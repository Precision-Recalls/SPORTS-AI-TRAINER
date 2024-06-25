import mediapipe as mp

from basketball_analytics.ball_analytics import classify_throw, calculate_throw_time
from basketball_analytics.player_analytics import get_pose_landmarks, calculate_jump_height, \
    calculate_distance_from_basket, calculate_player_level_while_throwing, count_steps

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

court_dimensions = {'width': 15, 'height': 28, 'scale_factor': 28 / 1.0}


def analyze_basketball_video(frames, hoop_position, backboard_y, court_dimensions):
    landmarks_sequence = [get_pose_landmarks(frame) for frame in frames]
    ball_positions = detect_ball(frames)

    steps = count_steps(landmarks_sequence)
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


video_analysis = analyze_basketball_video(frames, hoop_position, backboard_y, court_dimensions)
print(video_analysis)
