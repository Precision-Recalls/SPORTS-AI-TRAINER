import cv2
import numpy as np
import torch

# Initialize ball trajectory
ball_positions = []
prev_ball_position = None
ball_label = "basketball"  # Adjust this label based on your model's output


def track_ball(predictions, frame):
    global ball_positions, prev_ball_position
    current_ball_position = None
    min_distance = float('inf')

    for prediction in predictions:
        if prediction.class_name == ball_label:
            x = int(prediction.x)
            y = int(prediction.y)
            w = int(prediction.width)
            h = int(prediction.height)

            # Calculate center of the basketball
            center_x = int(x)
            center_y = int(y)
            ball_center = (center_x, center_y)

            # Draw a circle at the ball's center
            cv2.circle(frame, ball_center, 1, (0, 255, 0), -1)
            if prev_ball_position:
                distance = np.linalg.norm(np.array(ball_center) - np.array(prev_ball_position))
                if distance < min_distance:
                    min_distance = distance
                    current_ball_position = ball_center
            else:
                current_ball_position = ball_center
            print(f"Ball detected at: {ball_center}")  # Print ball coordinates
            cv2.putText(frame, f"Ball Coordinates: {ball_center}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 255), 2)
            break

    # Track the ball
    if current_ball_position:
        ball_positions.append(current_ball_position)
        prev_ball_position = current_ball_position
    elif prev_ball_position:
        ball_positions.append(prev_ball_position)

    # Draw the trajectory
    for i in range(1, len(ball_positions)):
        cv2.line(frame, ball_positions[i - 1], ball_positions[i], (0, 0, 255), 2)


def classify_throw(ball_positions, hoop_position, backboard_y):
    if not ball_positions:
        return "No throw detected"

    final_position = ball_positions[-1]
    x, y = final_position

    hoop_radius = 0.1  # Adjust based on your coordinate system

    if (x - hoop_position[0]) ** 2 + (y - hoop_position[1]) ** 2 < hoop_radius ** 2:
        return "Clean inside hoop"
    elif abs(x - hoop_position[0]) < hoop_radius and abs(y - hoop_position[1]) < hoop_radius:
        return "Inside hoop touching rim"
    elif y < backboard_y:
        return "Inside hoop reflecting from backboard"
    elif (x - hoop_position[0]) ** 2 + (y - hoop_position[1]) ** 2 < 1.5 * hoop_radius ** 2:
        return "On the rim"
    else:
        return "Outside the hoop"


def calculate_throw_time(ball_positions, frame_rate):
    if not ball_positions:
        return 0

    num_frames = len(ball_positions)
    throw_time = num_frames / frame_rate

    return throw_time
