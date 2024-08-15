import logging
import os
import sys
import tempfile

import cv2
import mediapipe as mp
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

from common.utils import write_video
from fitness_analytics.constants.features import Features, DerivedFeatures
from fitness_analytics.utils.df_utils import get_keypoint_df, get_keypoint_list, get_angle_from_keypoints_df, \
    identify_side_df, calculate_rep_df, calculate_derived_features_df, get_drill_bodyparts
from fitness_analytics.utils.draw_utils import draw_keypoints_and_connections
from fitness_analytics.utils.video_utils import return_frame_generator

mp_pose = mp.solutions.pose
logger = logging.Logger('ERROR')


def process(drill_name, video_blob_name, output_blob_name,
            azure_connection_string,
            azure_container_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
        container_client = blob_service_client.get_container_client(azure_container_name)

        video_blob_client = container_client.get_blob_client(video_blob_name)
        output_blob_client = container_client.get_blob_client(output_blob_name)

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
            video_blob_client.download_blob().readinto(temp_video_file)
            temp_video_file.seek(0)
            cap = cv2.VideoCapture(temp_video_file.name)

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        video_write = write_video(cap, output_blob_client)

        bodypart, complements, angle_range = get_drill_bodyparts(drill_name)

        # generate frames to be processed from the video
        frame_generator = return_frame_generator(cap)

        features = Features(name=drill_name, bodypart=bodypart, complements=complements)
        with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.60,
                min_tracking_confidence=0.60) as pose:

            for f, r in tqdm(frame_generator):
                results = pose.process(r)
                f.flags.writeable = True

                if results.pose_landmarks:
                    features.results.append(results)
                    draw_keypoints_and_connections(f, results)
                    video_write.write(f)

        df = get_keypoint_df([get_keypoint_list(result) for result in features.results])
        # df = perform_smoothning(df)

        features.side = identify_side_df(df, features.bodypart)
        derived_features = DerivedFeatures()
        bodypart_analysed = features.side + features.bodypart
        complements_analysed = [features.side + complement for complement in features.complements]
        df = get_angle_from_keypoints_df(df, bodypart_analysed, complements_analysed)
        df = calculate_rep_df(df)
        # creating response from features
        features.assign_features(df)

        # calculate derived features
        derived_features = calculate_derived_features_df(df, derived_features, angle_range, frame_rate)
        response = {
            'drill_features': features.return_json(),
            'derived_features': derived_features.return_json()
        }

        return response
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f'There is some issue with fitness process function at {exc_tb.tb_lineno}th line '
                     f'in {fname}, error {exc_type}')
    finally:
        cap.release()
        video_write.release()
        cv2.destroyAllWindows()
