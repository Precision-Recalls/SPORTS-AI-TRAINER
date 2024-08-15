from fitness_analytics.fitness_configs import STYLE, GAP, BODY_COLOR_BGR, VISIBILITY_THRESHOLD, PRESENCE_THRESHOLD
from fitness_analytics.fitness_configs import SKIP_FRAME
from fitness_analytics.constants.connections import Connections
from fitness_analytics.constants.body import Body
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
DrawingSpec = mp.solutions.drawing_utils.DrawingSpec


def get_pose_landmarks_to_draw(results):
    def get_value(x):
        try:
            return results.pose_landmarks.landmark[x]
        except Exception as e:
            None

    landmark_subset = landmark_pb2.NormalizedLandmarkList(
        landmark=[get_value(i) for i in Body.get_body_indexes()]
    )
    return landmark_subset


def drawline(img, pt1, pt2, color, thickness, style, gap):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def draw_connections(image, landmark_list, connections, connection_drawing_spec, style=STYLE, gap=GAP):
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < PRESENCE_THRESHOLD)):
            continue
        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                  image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    if connections:
        num_landmarks = len(landmark_list.landmark)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
            raise ValueError(f'Landmark index is out of range. Invalid connection '
                             f'from landmark #{start_idx} to landmark #{end_idx}.')

        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            drawing_spec = connection_drawing_spec
            drawline(image, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], drawing_spec.color,
                     drawing_spec.thickness, style=style, gap=gap)


def draw_keypoints_and_connections(f, results):
    mp_drawing.draw_landmarks(
        f, get_pose_landmarks_to_draw(results),
        landmark_drawing_spec=DrawingSpec(color=BODY_COLOR_BGR, thickness=5, circle_radius=5),
    )
    draw_connections(f, get_pose_landmarks_to_draw(results), Connections.values,
                     DrawingSpec(color=BODY_COLOR_BGR, thickness=2))


def draw_features(f, body_features, fps, skip_frame=SKIP_FRAME, feature_flag=True, angle_flag=True):
    writer_fps = int(fps / (skip_frame + 1))
    if feature_flag:
        cv2.putText(f, 'Total Reps: ' + str(body_features.total_reps),
                    (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1.1, (102, 255, 0), 2)
        cv2.putText(f, 'Time Elapsed: ' + str(round(len(body_features.angles) / writer_fps, 0)),
                    (20, 120), cv2.FONT_HERSHEY_COMPLEX, 1.1, (102, 255, 0), 2)
        cv2.putText(f, 'Direction: ' + str(body_features.directions[-1]),
                    (20, 160), cv2.FONT_HERSHEY_COMPLEX, 1.1, (102, 255, 0), 2)
    if angle_flag:
        cv2.putText(f, f'{body_features.body_part}: ' + str(body_features.angles[-1]) + ' deg',
                    (body_features.coords[-1][0], body_features.coords[-1][1] + 45), cv2.FONT_HERSHEY_COMPLEX, 1.1,
                    (102, 255, 0), 2)
