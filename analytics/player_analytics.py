import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def get_pose_landmarks(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None


def calculate_steps(landmarks_sequence):
    if not landmarks_sequence:
        return 0

    step_count = 0
    left_foot_positions = []
    right_foot_positions = []

    for landmarks in landmarks_sequence:
        left_foot_positions.append((landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y))
        right_foot_positions.append((landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y))

    for i in range(1, len(left_foot_positions)):
        if np.linalg.norm(np.array(left_foot_positions[i]) - np.array(left_foot_positions[i - 1])) > 0.1 or \
                np.linalg.norm(np.array(right_foot_positions[i]) - np.array(right_foot_positions[i - 1])) > 0.1:
            step_count += 1

    return step_count


def calculate_jump_height(landmarks_sequence):
    if not landmarks_sequence:
        return 0

    heel_positions = []

    for landmarks in landmarks_sequence:
        left_heel_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
        right_heel_y = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
        heel_positions.append((left_heel_y + right_heel_y) / 2)

    min_heel_y = min(heel_positions)
    max_heel_y = max(heel_positions)
    jump_height = min_heel_y - max_heel_y  # Assuming lower y value means higher position

    return jump_height


def calculate_distance_from_basket(player_position, basket_position, court_dimensions):
    player_x, player_y = player_position
    basket_x, basket_y = basket_position
    distance = np.sqrt((player_x - basket_x) ** 2 + (player_y - basket_y) ** 2) * court_dimensions['scale_factor']
    return distance


def calculate_player_level_while_throwing(landmarks_sequence):
    if not landmarks_sequence:
        return 0

    for landmarks in landmarks_sequence:
        if landmarks[mp_pose.PoseLandmark.LEFT_HAND.value]:
            left_hand_y = landmarks[mp_pose.PoseLandmark.LEFT_HAND.value].y
            return left_hand_y

    return 0