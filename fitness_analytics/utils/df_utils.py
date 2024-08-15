import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal

from fitness_analytics.constants.body import Body

from fitness_analytics.fitness_configs import SKIP_FRAME, WINDOW_LENGTH, POLYORDER, \
    RANGE_RATIO, WIDTH, HEIGHT, EXCERCISES

logger = logging.Logger('ERROR')


def get_drill_bodyparts(drill_name):
    excercise = EXCERCISES.get(drill_name)
    body_part = excercise.get('body_part', '')
    complements = excercise.get('complements')
    range = excercise.get('range', [])
    return body_part, complements, range


def get_keypoint_list(results):
    keypoints = []
    if not results.pose_landmarks:
        print('Human not detected')
    else:
        for datapoint in results.pose_landmarks.landmark:
            keypoints += [
                datapoint.x,
                datapoint.y,
                datapoint.z,
            ]
    return keypoints


def get_keypoint_df(list_of_keypoints):
    num_of_keypoints = int(len(list_of_keypoints[0]) / 3)
    return pd.DataFrame(list_of_keypoints,
                        columns=[f'{i}_{dim}' for i in range(num_of_keypoints) for dim in ['x', 'y', 'z']])


def perform_smoothning(keypoints_df):
    df = keypoints_df.copy()
    for i in df.columns: df[str(i)] = np.round(signal.savgol_filter(df[str(i)], WINDOW_LENGTH, POLYORDER), 2)
    return df


def identify_side_df(keypoints_df, bodypart):
    left_index = getattr(Body, 'left_' + bodypart)
    right_index = getattr(Body, 'right_' + bodypart)
    left_z = sum(keypoints_df[f'{left_index}_z']) / len(keypoints_df)
    right_z = sum(keypoints_df[f'{right_index}_z']) / len(keypoints_df)

    if left_z < right_z:
        return 'left_'
    else:
        return 'right_'


def get_angle_from_keypoints_df(keypoints_df, bodypart, complements, width=WIDTH, height=HEIGHT):
    def calculate_angle(x1, y1, z1, x2, y2, z2):
        ba = np.array([x1, y1])
        bc = np.array([x2, y2])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return round(np.degrees(angle), 2)

    def calculate_angle_with_z(x1, y1, z1, x2, y2, z2):
        ba = np.array([x1, y1, z1])
        bc = np.array([x2, y2, z2])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return round(np.degrees(angle), 2)

    df = keypoints_df.copy()
    b = getattr(Body, bodypart)
    a, c = [getattr(Body, complements[0]), getattr(Body, complements[1])]
    df['bax'] = df[f'{b}_x'] - df[f'{a}_x']
    df['bay'] = df[f'{b}_y'] - df[f'{a}_y']
    df['baz'] = df[f'{b}_z'] - df[f'{a}_z']
    df['bcx'] = df[f'{b}_x'] - df[f'{c}_x']
    df['bcy'] = df[f'{b}_y'] - df[f'{c}_y']
    df['bcz'] = df[f'{b}_z'] - df[f'{c}_z']

    df['angle'] = df.apply(lambda x: calculate_angle_with_z(x['bax'], x['bay'], x['baz'], x['bcx'], x['bcy'], x['bcz']),
                           axis=1)
    return df[['angle']]


def calculate_rep_df(df):
    def calculate_direction_change(diff, prev_diff):
        def is_nan(x):
            if x != x:
                return True
            else:
                return False

        if is_nan(diff) or is_nan(prev_diff) or diff * prev_diff >= 0 or diff * prev_diff < -0.5:
            return np.nan
        elif diff > 0:
            return 1
        else:
            return -1

    def calculate_pivot(x):
        if x['angle'] == x['group_max'] and x['boundary_angle'] == 1:
            return 1
        elif x['angle'] == x['group_min'] and x['boundary_angle'] == -1:
            return 1
        else:
            return 0

    drill_range = (df['angle'].min(), df['angle'].max())
    drill_max_threshold = drill_range[0] + (drill_range[1] - drill_range[0]) * RANGE_RATIO
    drill_min_threshold = drill_range[0] + (drill_range[1] - drill_range[0]) * (1 - RANGE_RATIO)
    df['boundary_angle'] = np.nan
    df.loc[df['angle'] > drill_max_threshold, 'boundary_angle'] = 1
    df.loc[df['angle'] < drill_min_threshold, 'boundary_angle'] = -1
    df['group_direction'] = df['boundary_angle'].values
    df['group_direction'] = df['group_direction'].ffill(limit=200)
    df['prev_group_direction'] = list([np.nan] + list(df['group_direction'].values[:-1]))
    df['group_change'] = df.apply(lambda x: 1 if x['prev_group_direction'] * x['group_direction'] < 0 else 0, axis=1)
    df['group_id'] = df['group_change'].cumsum()
    df['group_max'] = df.groupby('group_id')['angle'].transform('max')
    df['group_min'] = df.groupby('group_id')['angle'].transform('min')
    df['pivot_flag'] = df.apply(lambda x: calculate_pivot(x), axis=1)
    df['direction'] = df['pivot_flag'].cumsum()
    df['direction'] = df['direction'].apply(lambda x: x - 1 if x != 0 else 0)
    df['rep'] = df['direction'].apply(lambda x: int(x / 2) + 1)
    df.drop(
        ['group_change', 'group_direction', 'prev_group_direction', 'pivot_flag', 'group_min', 'group_max', 'group_id',
         'boundary_angle'], axis=1, inplace=True)
    return df.reset_index(drop=True)


def calculate_derived_features_df(df, derived_features, angle_range, fps):
    try:
        def calculate_time_from_frames(num_of_frames, fps, skip_frames=SKIP_FRAME):
            epsilon = 0.0001
            return round((num_of_frames * (skip_frames + 1)) / (fps - epsilon), 1)

        def calculate_completion(min_angle, max_angle, min_range, max_range):
            if max_range and min_range:
                completion_ratio = (min(max_angle, max_range) - max(min_angle, min_range)) / (max_range - min_range)
                return round(completion_ratio * 100, 2)
            else:
                return -1

        MIN_RANGE, MAX_RANGE = angle_range if len(angle_range) == 2 else (None, None)
        f = df.groupby('rep').agg({'angle': ["min", "max"], 'direction': 'count'}).reset_index()
        f.columns = ['rep', 'min_angle', 'max_angle', 'rep_frames']
        f.set_index('rep', inplace=True)
        for rep in f.index:
            min_angle, max_angle, rep_frames = f.loc[rep, :].values
            derived_features.min_angles[rep] = min_angle
            derived_features.max_angles[rep] = max_angle
            derived_features.rep_time[rep] = calculate_time_from_frames(rep_frames, fps)
            derived_features.rep_completion[rep] = calculate_completion(min_angle, max_angle, MIN_RANGE, MAX_RANGE)
        derived_features.total_reps = int(max(derived_features.rep_time.keys()))
        derived_features.total_time = int(sum(derived_features.rep_time.values()))
        return derived_features
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f'There is some issue with calculate_derived_features_df at {exc_tb.tb_lineno}th line '
                     f'in {fname}, error {exc_type}')
