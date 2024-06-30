from common.utils import calculate_angle


class Player:
    global prev_left_ankle_y, prev_right_ankle_y, wait_frames
    # prev_left_ankle_y = None
    # #prev_right_ankle_y = None
    # wait_frames = 0

    def __init__(self, frame, body_index):
        self.frame = frame
        self.body_index = body_index
        # Initialize counters and previous positions
        self.prev_left_ankle_y = None
        self.prev_right_ankle_y = None
        self.step_threshold = 12
        self.min_wait_frames = 8

        self.prev_right_ankle_y = prev_right_ankle_y
        self.prev_left_ankle_y = prev_left_ankle_y
        self.wait_frames = wait_frames

    def count_steps(self, rounded_pose_results):
        steps = 0

        # Get the key points for the body parts
        try:
            left_knee = rounded_pose_results[0][self.body_index["left_knee"]]
            right_knee = rounded_pose_results[0][self.body_index["right_knee"]]
            left_ankle = rounded_pose_results[0][self.body_index["left_ankle"]]
            right_ankle = rounded_pose_results[0][self.body_index["right_ankle"]]
            keypoints = [left_knee, right_knee, left_ankle, right_ankle]

            if all(point[2] > 0.5 for point in keypoints):
                if self.prev_left_ankle_y is not None and self.prev_right_ankle_y is not None and self.wait_frames == 0:
                    left_diff = abs(left_ankle[1] - self.prev_left_ankle_y)
                    right_diff = abs(right_ankle[1] - self.prev_right_ankle_y)

                    if max(left_diff, right_diff) > self.step_threshold:
                        steps += 1
                        self.wait_frames = self.min_wait_frames
                self.prev_left_ankle_y = left_ankle[1]
                self.prev_right_ankle_y = right_ankle[1]

                if self.wait_frames > 0:
                    self.wait_frames -= 1
            return steps
        except Exception as e:
            print(f"There is some error in steps counting :- {e}")

    def calculate_elbow_angles(self, rounded_pose_results):
        """
        Calculate the elbow angles for both left and right elbows.
        """
        angles = {}
        try:
            left_shoulder = rounded_pose_results[0][self.body_index["left_shoulder"]][:2]
            left_elbow = rounded_pose_results[0][self.body_index["left_elbow"]][:2]
            left_wrist = rounded_pose_results[0][self.body_index["left_wrist"]][:2]
            right_shoulder = rounded_pose_results[0][self.body_index["right_shoulder"]][:2]
            right_elbow = rounded_pose_results[0][self.body_index["right_elbow"]][:2]
            right_wrist = rounded_pose_results[0][self.body_index["right_wrist"]][:2]

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            angles["left_elbow"] = left_elbow_angle
            angles["right_elbow"] = right_elbow_angle
            return angles
        except Exception as e:
            print("Unable to calculate elbow angles.")
