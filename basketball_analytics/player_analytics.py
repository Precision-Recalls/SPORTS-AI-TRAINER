import cv2
#import mediapipe as mp
import numpy as np

from utils import calculate_angle

# Define the body part indices
body_index = {"left_shoulder": 5, "left_elbow": 7, "left_wrist": 9,
              "right_shoulder": 6, "right_elbow": 8, "right_wrist": 10,
              "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}

# Initialize counters and previous positions
step_counter = 0
prev_left_ankle_y = None
prev_right_ankle_y = None
step_threshold = 12
min_wait_frames = 8
wait_frames = 0


def count_steps(pose_results):
    global prev_left_ankle_y, prev_right_ankle_y, wait_frames
    steps = 0

    # Round the results to the nearest decimal
    rounded_results = np.round(pose_results.keypoints.data.numpy(), 1)

    # Get the keypoints for the body parts
    try:
        left_knee = rounded_results[0][body_index["left_knee"]]
        right_knee = rounded_results[0][body_index["right_knee"]]
        left_ankle = rounded_results[0][body_index["left_ankle"]]
        right_ankle = rounded_results[0][body_index["right_ankle"]]

        if (
                (left_knee[2] > 0.5)
                and (right_knee[2] > 0.5)
                and (left_ankle[2] > 0.5)
                and (right_ankle[2] > 0.5)
        ):
            if (
                    prev_left_ankle_y is not None
                    and prev_right_ankle_y is not None
                    and wait_frames == 0
            ):
                left_diff = abs(left_ankle[1] - prev_left_ankle_y)
                right_diff = abs(right_ankle[1] - prev_right_ankle_y)

                if max(left_diff, right_diff) > step_threshold:
                    steps += 1
                    wait_frames = min_wait_frames

            prev_left_ankle_y = left_ankle[1]
            prev_right_ankle_y = right_ankle[1]

            if wait_frames > 0:
                wait_frames -= 1

    except:
        print("No human detected.")

    return steps




def draw_pose_landmarks_and_bboxes(frame, pose_results):
    """
    Draw pose landmarks (keypoints) and bounding boxes on the frame.
    """
    rounded_results = np.round(pose_results.keypoints.data.numpy(), 1)

    for person in rounded_results:
        keypoints = person[:, :2]
        confidences = person[:, 2]

        # Filter keypoints by confidence threshold
        valid_keypoints = keypoints[confidences > 0.5]
        if len(valid_keypoints) > 0:
            min_x = int(np.min(valid_keypoints[:, 0]))
            max_x = int(np.max(valid_keypoints[:, 0]))
            min_y = int(np.min(valid_keypoints[:, 1]))
            max_y = int(np.max(valid_keypoints[:, 1]))

            # Draw bounding box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw keypoints
            for idx, keypoint in enumerate(valid_keypoints):
                center = (int(keypoint[0]), int(keypoint[1]))
                cv2.circle(frame, center, 3, (255, 0, 0), -1)
                # cv2.putText(frame, f"{idx}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def calculate_elbow_angles(pose_results):
    """
    Calculate the elbow angles for both left and right elbows.
    """
    angles = {}
    rounded_results = np.round(pose_results.keypoints.data.numpy(), 1)

    try:
        left_shoulder = rounded_results[0][body_index["left_shoulder"]][:2]
        left_elbow = rounded_results[0][body_index["left_elbow"]][:2]
        left_wrist = rounded_results[0][body_index["left_wrist"]][:2]
        right_shoulder = rounded_results[0][body_index["right_shoulder"]][:2]
        right_elbow = rounded_results[0][body_index["right_elbow"]][:2]
        right_wrist = rounded_results[0][body_index["right_wrist"]][:2]

        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        angles["left_elbow"] = left_elbow_angle
        angles["right_elbow"] = right_elbow_angle

    except:
        print("Unable to calculate elbow angles.")
    return angles
