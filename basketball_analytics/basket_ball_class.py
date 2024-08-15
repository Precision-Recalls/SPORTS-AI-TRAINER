import cv2
import numpy as np

import logging
import tempfile
from azure.storage.blob import BlobServiceClient
from basketball_analytics.player_class import Player
from basketball_analytics.shot_detector_class import ShotDetector

from common.utils import scale_text, video_writer

logger = logging.Logger('ERROR')


class BasketBallGame:
    def __init__(self, model, pose_model, class_names, video_blob_name,
                 output_blob_name, body_index, azure_connection_string, azure_container_name):
        self.model = model
        self.pose_model = pose_model
        self.class_names = class_names
        self.video_blob_name = video_blob_name
        self.output_blob_name = output_blob_name
        self.body_index = body_index
        self.player = Player(self.body_index)
        self.frame_count = 0
        self.frame = None
        self.frame_steps = []
        self.all_shot_data = []

        self.blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
        container_client = self.blob_service_client.get_container_client(azure_container_name)

        self.video_blob_client = container_client.get_blob_client(self.video_blob_name)
        self.output_blob_client = container_client.get_blob_client(self.output_blob_name)

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
            self.video_blob_client.download_blob().readinto(temp_video_file)
            temp_video_file.seek(0)
            self.cap = cv2.VideoCapture(temp_video_file.name)

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.shot_detector = ShotDetector(self.class_names, self.body_index, self.frame_rate)
        self.video_writer = video_writer(self.cap, self.output_blob_client)
        self.run()

    def run(self):
        try:
            while True:
                ret, self.frame = self.cap.read()
                if not ret:
                    break

                object_detection_results = self.model(self.frame, conf=0.7, iou=0.4, stream=True)
                pose_results = self.pose_model(self.frame, verbose=False, conf=0.7)
                step_counter = 0
                elbow_angles = {}
                if pose_results:
                    rounded_pose_results = np.round(pose_results[0].keypoints.data.numpy(), 1)
                    steps = self.player.count_steps(rounded_pose_results)
                    step_counter += steps
                    elbow_angles = self.player.calculate_elbow_angles(rounded_pose_results)
                    text, position, font_scale, thickness = scale_text(self.frame, f"Total Steps Taken: {step_counter}",
                                                                       (10, 75), 1, 2)

                    cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

                if object_detection_results:
                    self.frame, shot_attempt_data = self.shot_detector.run(
                        self.frame_count,
                        self.frame,
                        step_counter,
                        object_detection_results,
                        pose_results
                    )
                    if shot_attempt_data:
                        self.all_shot_data.append(shot_attempt_data)

                self.frame_count += 1
                self.video_writer.write(self.frame)
                # Close if 'q' is clicked
                if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                    break
        except Exception as e:
            logger.error(f'There is some issue with basketball video file processing: {e}')
        finally:
            self.cap.release()
            self.video_writer.release()
            cv2.destroyAllWindows()

    def to_list(self):
        return self.all_shot_data

    def __iter__(self):
        return iter(self.to_list())

    def __getitem__(self, index):
        return self.to_list()[index]

    def __len__(self):
        return len(self.to_list())
