# import logging
# import os
# import sys
# import tempfile

# import cv2
# import mediapipe as mp
# from azure.storage.blob import BlobServiceClient
# from tqdm import tqdm

# from common.utils import write_video
# from fitness_analytics.constants.features import Features, DerivedFeatures
# from fitness_analytics.utils.df_utils import get_keypoint_df, get_keypoint_list, get_angle_from_keypoints_df, \
#     identify_side_df, calculate_rep_df, calculate_derived_features_df, get_drill_bodyparts
# from fitness_analytics.utils.draw_utils import draw_keypoints_and_connections
# from fitness_analytics.utils.video_utils import return_frame_generator

# mp_pose = mp.solutions.pose
# logger = logging.Logger('ERROR')


# def process(drill_name, video_blob_name, output_blob_name,
#             azure_connection_string,
#             azure_container_name):
#     try:
#         blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
#         container_client = blob_service_client.get_container_client(azure_container_name)

#         video_blob_client = container_client.get_blob_client(video_blob_name)
#         output_blob_client = container_client.get_blob_client(output_blob_name)

#         with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
#             video_blob_client.download_blob().readinto(temp_video_file)
#             temp_video_file.seek(0)
#             cap = cv2.VideoCapture(temp_video_file.name)

#         frame_rate = cap.get(cv2.CAP_PROP_FPS)
#         video_write = write_video(cap, output_blob_client)

#         bodypart, complements, angle_range = get_drill_bodyparts(drill_name)

#         # generate frames to be processed from the video
#         frame_generator = return_frame_generator(cap)

#         features = Features(name=drill_name, bodypart=bodypart, complements=complements)
#         with mp_pose.Pose(
#                 static_image_mode=False,
#                 model_complexity=1,
#                 smooth_landmarks=True,
#                 min_detection_confidence=0.60,
#                 min_tracking_confidence=0.60) as pose:

#             for f, r in tqdm(frame_generator):
#                 results = pose.process(r)
#                 f.flags.writeable = True

#                 if results.pose_landmarks:
#                     features.results.append(results)
#                     draw_keypoints_and_connections(f, results)
#                     video_write.write(f)

#         df = get_keypoint_df([get_keypoint_list(result) for result in features.results])
#         # df = perform_smoothning(df)

#         features.side = identify_side_df(df, features.bodypart)
#         derived_features = DerivedFeatures()
#         bodypart_analysed = features.side + features.bodypart
#         complements_analysed = [features.side + complement for complement in features.complements]
#         df = get_angle_from_keypoints_df(df, bodypart_analysed, complements_analysed)
#         df = calculate_rep_df(df)
#         # creating response from features
#         features.assign_features(df)

#         # calculate derived features
#         derived_features = calculate_derived_features_df(df, derived_features, angle_range, frame_rate)
#         response = {
#             'drill_features': features.return_json(),
#             'derived_features': derived_features.return_json()
#         }

#         return response
#     except Exception as e:
#         exc_type, exc_obj, exc_tb = sys.exc_info()
#         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#         logger.error(f'There is some issue with fitness process function at {exc_tb.tb_lineno}th line '
#                      f'in {fname}, error {exc_type}')
#     finally:
#         cap.release()
#         video_write.release()
#         cv2.destroyAllWindows()
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
from fitness_analytics.utils.draw_utils import draw_keypoints_and_connections, draw_features
from fitness_analytics.utils.video_utils import return_frame_generator

mp_pose = mp.solutions.pose
logger = logging.Logger('ERROR')


class Fitness:
    def __init__(self, drill_name, video_blob_name, output_blob_name,
                 azure_connection_string, azure_container_name):
        self.drill_name = drill_name
        self.video_blob_name = video_blob_name
        self.output_blob_name = output_blob_name
        self.azure_connection_string = azure_connection_string
        self.azure_container_name = azure_container_name
        self.response = {}  
        self.blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
        container_client = self.blob_service_client.get_container_client(azure_container_name)
        self.video_blob_client = container_client.get_blob_client(video_blob_name)
        self.output_blob_client = container_client.get_blob_client(output_blob_name)

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
            self.video_blob_client.download_blob().readinto(temp_video_file)
            temp_video_file.seek(0)
            self.cap = cv2.VideoCapture(temp_video_file.name)

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_write = write_video(self.cap, self.output_blob_client)

        self.bodypart, self.complements, self.angle_range = get_drill_bodyparts(drill_name)
        self.features = Features(name=drill_name, bodypart=self.bodypart, complements=self.complements)
        self.derived_features = DerivedFeatures()

        self.run()

    def run(self):
        try:
            #frame_generator = return_frame_generator(self.cap)
            while True:
                ret, self.frame = self.cap.read()
                if not ret:
                    # End o
                    break
                with mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        min_detection_confidence=0.60,
                        min_tracking_confidence=0.60) as pose:

                    # for f, r in tqdm(frame_generator):
                    #     results = pose.process(r)
                    #     f.flags.writeable = True
                    results = pose.process(self.frame)
                    self.frame.flags.writeable = True

                    if results.pose_landmarks:
                        self.features.results.append(results)
                        draw_keypoints_and_connections(self.frame, results)
                        self.video_write.write(self.frame)

                if self.features.results:
                    df = get_keypoint_df([get_keypoint_list(result) for result in self.features.results])
                    self.features.side = identify_side_df(df, self.features.bodypart)
                    bodypart_analysed = self.features.side + self.features.bodypart
                    complements_analysed = [self.features.side + complement for complement in self.features.complements]
                    df = get_angle_from_keypoints_df(df, bodypart_analysed, complements_analysed)
                    df = calculate_rep_df(df)
                    # creating response from features
                    self.features.assign_features(df)

                    # calculate derived features
                    self.derived_features = calculate_derived_features_df(df, self.derived_features, self.angle_range, self.frame_rate)
                    draw_features(self.frame, self.derived_features, self.frame_rate,skip_frame=3, feature_flag=True, angle_flag=True)
                    self.video_write.write(self.frame)
                    self.response = {
                        'drill_features': self.features.return_json(),
                        'derived_features': self.derived_features.return_json()
                    }
                    draw_features(self.frame, self.derived_features, self.frame_rate,skip_frame=3, feature_flag=True, angle_flag=True)
                    self.video_write.write(self.frame)
                else:
                    logger.info("No pose landmarks detected in this frame.")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f'There is some issue with fitness process function at {exc_tb.tb_lineno}th line '
                         f'in {fname}, error {exc_type}')
        finally:
            self.cap.release()
            self.video_write.release()
            cv2.destroyAllWindows()

    def to_dict(self):
        return self.response

    def __iter__(self):
        return iter(self.to_dict())

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __len__(self):
        return len(self.to_dict())
