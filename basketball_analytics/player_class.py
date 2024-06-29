import numpy as np
from common.utils import calculate_angle


class Player:
    def __init__(self, frame, pose_results, body_index):
        # Load the YOLO model created from main.py - change text to your relative path
        self.pose_results = pose_results
        self.frame = frame
        self.body_index = body_index
        # Initialize counters and previous positions
        self.prev_left_ankle_y = None
        self.prev_right_ankle_y = None
        self.step_threshold = 12
        self.min_wait_frames = 8

    def count_steps(self):
        #global prev_left_ankle_y, prev_right_ankle_y, wait_frames
        steps = 0
        wait_frames = 0

        # Round the results to the nearest decimal
        rounded_results = np.round(self.pose_results.keypoints.data.numpy(), 1)

        # Get the keypoints for the body parts
        try:
            left_knee = rounded_results[0][self.body_index["left_knee"]]
            right_knee = rounded_results[0][self.body_index["right_knee"]]
            left_ankle = rounded_results[0][self.body_index["left_ankle"]]
            right_ankle = rounded_results[0][self.body_index["right_ankle"]]

            if (
                    (left_knee[2] > 0.5)
                    and (right_knee[2] > 0.5)
                    and (left_ankle[2] > 0.5)
                    and (right_ankle[2] > 0.5)
            ):
                if (
                        self.prev_left_ankle_y is not None
                        and self.prev_right_ankle_y is not None
                        and wait_frames == 0
                ):
                    left_diff = abs(left_ankle[1] - self.prev_left_ankle_y)
                    right_diff = abs(right_ankle[1] - self.prev_right_ankle_y)

                    if max(left_diff, right_diff) > self.step_threshold:
                        steps += 1
                        wait_frames = self.min_wait_frames

                self.prev_left_ankle_y = left_ankle[1]
                self.prev_right_ankle_y = right_ankle[1]

                if wait_frames > 0:
                    wait_frames -= 1
            return steps
        except Exception as e:
            print("No human detected.")
        

    def calculate_elbow_angles(self):
        """
        Calculate the elbow angles for both left and right elbows.
        """
        angles = {}
        rounded_results = np.round(self.pose_results.keypoints.data.numpy(), 1)
        try:
            left_shoulder = rounded_results[0][self.body_index["left_shoulder"]][:2]
            left_elbow = rounded_results[0][self.body_index["left_elbow"]][:2]
            left_wrist = rounded_results[0][self.body_index["left_wrist"]][:2]
            right_shoulder = rounded_results[0][self.body_index["right_shoulder"]][:2]
            right_elbow = rounded_results[0][self.body_index["right_elbow"]][:2]
            right_wrist = rounded_results[0][self.body_index["right_wrist"]][:2]

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            angles["left_elbow"] = left_elbow_angle
            angles["right_elbow"] = right_elbow_angle
            return angles
        except Exception as e:
            print("Unable to calculate elbow angles.")
    