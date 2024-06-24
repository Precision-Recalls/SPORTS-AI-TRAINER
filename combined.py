import torch
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Frame Extraction: The extract_frames function handles both video input and live feed.
# Hoop and Backboard Positions: Defined as hoop_position and backboard_y.
# Landmark Detection: Using MediaPipe to get pose landmarks for each frame.
# Ball Detection: Using YOLOv5 to detect the basketball in each frame.
# Steps Calculation: Based on the movement of foot landmarks.
# Jump Height Calculation: Based on the vertical movement of heel landmarks.
# Distance Calculation: Between the player and the hoop, scaled to court dimensions.
# Player Level Calculation: Vertical position of hand landmarks while throwing.
# Throw Classification: Determines the outcome of the throw based on ball's final position relative to the hoop.
# Throw Time Calculation: Calculates the time taken for the throw based on frame count and frame rate.
    

def get_pose_landmarks(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None

def detect_ball(frames):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    ball_positions = []

    for frame in frames:
        results = model(frame)
        for *xyxy, conf, cls in results.xyxy[0]:
            if cls == 0:  # Assuming class 0 is the basketball
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                ball_positions.append((x_center, y_center))

    return ball_positions

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
        if np.linalg.norm(np.array(left_foot_positions[i]) - np.array(left_foot_positions[i-1])) > 0.1 or \
           np.linalg.norm(np.array(right_foot_positions[i]) - np.array(right_foot_positions[i-1])) > 0.1:
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
    distance = np.sqrt((player_x - basket_x)**2 + (player_y - basket_y)**2) * court_dimensions['scale_factor']
    return distance

def calculate_player_level_while_throwing(landmarks_sequence):
    if not landmarks_sequence:
        return 0

    for landmarks in landmarks_sequence:
        if landmarks[mp_pose.PoseLandmark.LEFT_HAND.value]:
            left_hand_y = landmarks[mp_pose.PoseLandmark.LEFT_HAND.value].y
            return left_hand_y

    return 0

def classify_throw(ball_positions, hoop_position, backboard_y):
    if not ball_positions:
        return "No throw detected"

    final_position = ball_positions[-1]
    x, y = final_position

    hoop_radius = 0.1  # Adjust based on your coordinate system

    if (x - hoop_position[0])**2 + (y - hoop_position[1])**2 < hoop_radius**2:
        return "Clean inside hoop"
    elif abs(x - hoop_position[0]) < hoop_radius and abs(y - hoop_position[1]) < hoop_radius:
        return "Inside hoop touching rim"
    elif y < backboard_y:
        return "Inside hoop reflecting from backboard"
    elif (x - hoop_position[0])**2 + (y - hoop_position[1])**2 < 1.5 * hoop_radius**2:
        return "On the rim"
    else:
        return "Outside the hoop"

def calculate_throw_time(ball_positions, frame_rate):
    if not ball_positions:
        return 0

    num_frames = len(ball_positions)
    throw_time = num_frames / frame_rate

    return throw_time

def analyze_basketball_video(frames, hoop_position, backboard_y, court_dimensions):
    landmarks_sequence = [get_pose_landmarks(frame) for frame in frames]
    ball_positions = detect_ball(frames)
    
    steps = calculate_steps(landmarks_sequence)
    jump_height = calculate_jump_height(landmarks_sequence)
    player_position = (landmarks_sequence[-1][mp_pose.PoseLandmark.LEFT_HEEL.value].x, 
                       landmarks_sequence[-1][mp_pose.PoseLandmark.LEFT_HEEL.value].y)
    distance_from_basket = calculate_distance_from_basket(player_position, hoop_position, court_dimensions)
    player_level_while_throwing = calculate_player_level_while_throwing(landmarks_sequence)
    throw_classification = classify_throw(ball_positions, hoop_position, backboard_y)
    frame_rate = 30  # Assume 30 frames per second
    throw_time = calculate_throw_time(ball_positions, frame_rate)

    return {
        'steps': steps,
        'jump_height': jump_height,
        'distance_from_basket': distance_from_basket,
        'player_level_while_throwing': player_level_while_throwing,
        'throw_classification': throw_classification,
        'throw_time': throw_time
    }

# Usage
court_dimensions = {'width': 15, 'height': 28, 'scale_factor': 28 / 1.0}
video_analysis = analyze_basketball_video(frames, hoop_position, backboard_y, court_dimensions)
print(video_analysis)
