import cv2
import numpy as np
from ultralytics import YOLO
from basketball_analytics.player_analytics import draw_pose_landmarks_and_bboxes, calculate_elbow_angles, count_steps

# from utils import display_angles
# from utils import scale_text

model = YOLO(r"C:\Users\Abhay\PycharmProjects\SPORTS-AI-TRAINER\best_yolo8s.pt")
pose_model = YOLO(
    r'C:\Users\Abhay\PycharmProjects\SPORTS-AI-TRAINER\assets\models\yolov8s-pose.pt')  # Assuming you have a YOLO pose model
# Initialize counters
step_counter = 0
ball_positions = []
prev_ball_position = None
shot_detected = False
shot_counter = 0
goal_counter = 0

shot_frame_counter = 0
goal_frame_counter = 0
min_frames_between_shots = 30  # Minimum frames between shots
min_frames_between_goals = 60  # Minimum frames between goals

# Labels
ball_label = "ball"  # Adjust this label based on your model's output
basket_label = "basket"


def scale_text(frame, text, position, font_scale, thickness):
    """Scale text size and position according to frame size."""
    frame_height, frame_width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_scale * min(frame_width, frame_height) / 1000  # Scale based on frame size
    thickness = int(thickness * min(frame_width, frame_height) / 1000)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    # position = (int(position[0] * frame_width), int(position[1] * frame_height))
    return text, position, font_scale, thickness


def display_angles(frame, angles):
    """
    Display the calculated elbow angles on the frame.
    """
    if "left_elbow" in angles:
        text, position, font_scale, thickness = scale_text(frame, f"Left Elbow Angle: {angles['left_elbow']:.2f}",
                                                           (10, 60), 1, 2)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    if "right_elbow" in angles:
        text, position, font_scale, thickness = scale_text(frame, f"Right Elbow Angle: {angles['right_elbow']:.2f}",
                                                           (10, 90), 1, 2)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


def calculate_angle(a, b, c):
    try:
        """
        Calculate the angle between three points.
        """
        ab = np.array([b[0] - a[0], b[1] - a[1]])
        cb = np.array([b[0] - c[0], b[1] - c[1]])
        cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except Exception as e:
        print(f"There is some issue with angle calculation:- {e}")


def track_ball(predictions, frame):
    global ball_positions, prev_ball_position
    current_ball_position = None
    min_distance = float('inf')

    for bbox in predictions:
        x, y, w, h, conf, cls = bbox[:6]
        if model.names[int(cls)] == ball_label:
            center_x = int(x)
            center_y = int(y)
            ball_center = (center_x, center_y)

            # Draw a circle at the ball's center
            cv2.circle(frame, ball_center, 5, (0, 255, 0), -1)
            if prev_ball_position:
                distance = np.linalg.norm(np.array(ball_center) - np.array(prev_ball_position))
                if distance < min_distance:
                    min_distance = distance
                    current_ball_position = ball_center
            else:
                current_ball_position = ball_center
            print(f"Ball detected at: {ball_center}")  # Print ball coordinates
            # Scale text
            text, position, font_scale, thickness = scale_text(frame, f"Ball Coordinates: {ball_center}", (10, 120), 1,
                                                               2)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            break

    # Track the ball
    if current_ball_position:
        ball_positions.append(current_ball_position)
        prev_ball_position = current_ball_position
    elif prev_ball_position:
        ball_positions.append(prev_ball_position)

    # Draw the trajectory with smoothing
    if len(ball_positions) > 1:
        smoothed_positions = []
        for i in range(1, len(ball_positions)):
            smoothed_position = (
                int((ball_positions[i - 1][0] + ball_positions[i][0]) / 2),
                int((ball_positions[i - 1][1] + ball_positions[i][1]) / 2)
            )
            smoothed_positions.append(smoothed_position)
        # for i in range(1, len(smoothed_positions)):
        #   cv2.line(frame, smoothed_positions[i - 1], smoothed_positions[i], (0, 0, 255), 2)


# The detect_shot_and_goal function uses debounce logic to avoid false positives
def detect_shot_and_goal(ball_positions, basket_bbox, frame):
    global shot_detected, shot_counter, goal_counter
    global shot_frame_counter, goal_frame_counter

    basket_center = ((basket_bbox[0] + basket_bbox[2]) // 2, (basket_bbox[1] + basket_bbox[3]) // 2)
    proximity_radius = 50  # Adjust proximity radius as needed

    if len(ball_positions) < 2:
        return

    for pos in ball_positions:
        if np.linalg.norm(np.array(pos) - np.array(
                basket_center)) <= proximity_radius and shot_frame_counter > min_frames_between_shots:
            shot_detected = True
            shot_counter += 1
            shot_frame_counter = 0
            # text, position, font_scale, thickness = scale_text(frame, f"Shot Detected! Count: {shot_counter}", (10,
            # 150), 1, 3) cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255),
            # thickness)
            break

    if shot_detected:
        basket_bottom = basket_bbox[1] - 10
        basket_top = basket_bbox[3] + 10
        for i in range(1, len(ball_positions)):
            if ball_positions[i - 1][1] < basket_top and ball_positions[i][
                1] > basket_bottom and goal_frame_counter > min_frames_between_goals:
                goal_counter += 1
                goal_frame_counter = 0
                # text, position, font_scale, thickness = scale_text(frame, f"Goal! Count: {goal_counter}", (10, 170), 1, 3)
                # cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
                shot_detected = False
                break


def process_frame(frame):
    global step_counter, prev_left_ankle_y, prev_right_ankle_y, wait_frames
    global ball_positions, prev_ball_position
    global shot_frame_counter, goal_frame_counter
    # Run object detection inference on the frame
    results = model(frame, conf=0.7, iou=0.4)
    boxes = results[0].boxes.xywh  # Extracting bounding boxes in [x, y, w, h] format
    confs = results[0].boxes.conf  # Extracting confidence scores
    classes = results[0].boxes.cls  # Extracting class indices

    # Run pose estimation on the frame
    pose_results = pose_model(frame, verbose=False, conf=0.7)[0]
    steps = count_steps(pose_results)
    step_counter += steps
    # Combine the extracted information into a single array for easier processing
    predictions = np.hstack((boxes, confs.reshape(-1, 1), classes.reshape(-1, 1)))

    basket_bbox = None
    for bbox in predictions:
        x, y, w, h, conf, cls = bbox
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        label = model.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if label == basket_label:
            basket_bbox = (x1, y1, x2, y2)

    elbow_angles = calculate_elbow_angles(pose_results)
    display_angles(frame, elbow_angles)

    # Annotate the frame with the step count
    # cv2.putText(frame, f"Steps: {step_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    text, position, font_scale, thickness = scale_text(frame, f"Steps: {step_counter}", (10, 30), 1, 2)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Track ball trajectory
    track_ball(predictions, frame)
    if basket_bbox:
        detect_shot_and_goal(ball_positions, basket_bbox, frame)

    # Increment frame counters
    shot_frame_counter += 1
    goal_frame_counter += 1

    # Display shot and goal counters
    text, position, font_scale, thickness = scale_text(frame, f"Shots: {shot_counter}", (10, 150), 1, 3)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    text, position, font_scale, thickness = scale_text(frame, f"Goals: {goal_counter}", (10, 180), 1, 3)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return frame


def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return

    # Process and display the image
    annotated_image = process_frame(image)
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path):
    # Open the video capture
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Could not open or find the video: {video_path}")
        return

    # Process each frame in the video
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        annotated_frame = process_frame(frame)

        # Display the annotated frame
        cv2.imshow('Annotated Video', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # image_path = "test/sample_image.jpg"
    # process_image(image_path)

    # Process a video
    video_path = r"C:\Users\Abhay\PycharmProjects\SPORTS-AI-TRAINER\test\two_score_two_miss.mp4"
    process_video(video_path)
