import cv2
import numpy as np

from basketball_analytics.player_class import Player
from basketball_analytics.shot_detector_class import ShotDetector
from common.utils import display_angles, scale_text, video_writer


class BasketBallGame:
    def __init__(self, model, pose_model, class_names, video_link, output_video_path, body_index):
        # Essential variables
        self.model = model
        self.pose_model = pose_model
        self.class_names = class_names
        self.cap = cv2.VideoCapture(video_link)
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.body_index = body_index
        self.player = Player(self.body_index)
        self.shot_detector = ShotDetector(self.class_names, self.body_index, self.frame_rate)
        self.frame_count = 0
        self.frame = None
        self.frame_steps = []
        self.video_writer = video_writer(self.cap, output_video_path)
        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                # End of the video or an error occurred
                break

            object_detection_results = self.model(self.frame, conf=0.7, iou=0.4, stream=True)
            pose_results = self.pose_model(self.frame, verbose=False, conf=0.7)
            step_counter = 0
            if pose_results:
                # Round the results to the nearest decimal
                rounded_pose_results = np.round(pose_results[0].keypoints.data.numpy(), 1)
                steps = self.player.count_steps(rounded_pose_results)
                step_counter += steps
                # steps_before_shot += steps
                elbow_angles = self.player.calculate_elbow_angles(rounded_pose_results)
                if elbow_angles:
                    display_angles(self.frame, elbow_angles)

                # Annotate the frame with the step count
                text, position, font_scale, thickness = scale_text(self.frame, f"Total Steps Taken: {step_counter}",
                                                                   (10, 75), 1, 2)
                cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            if object_detection_results:
                self.frame = self.shot_detector.run(
                    self.frame_count,
                    self.frame,
                    step_counter,
                    object_detection_results,
                    pose_results
                )
            self.frame_count += 1
            self.video_writer.write(self.frame)
            # cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
