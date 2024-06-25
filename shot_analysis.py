import cv2
import numpy as np
import supervision as sv
from inference import get_model

# Initialize the object detection and pose estimation models
model = get_model(model_id="tracer-basketball/3")
# pose_model = YOLO("AI-Basketball-Referee/yolov8s-pose.pt") # Assuming you have a YOLO pose model

# Create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Initialize ball trajectory
ball_positions = []
prev_ball_position = None
ball_label = "basketball"  # Adjust this label based on your model's output

# Initialize hoop position
hoop_positions = []
hoop_label = "rim"  # Adjust this label based on your model's output

# Initialize counters and state variables
shot_counter = 0
goal_counter = 0
cooldown_frames = 30  # Cooldown period to avoid multiple detections in quick succession
cooldown_timer = 0
ball_in_hoop = False


def detect_shot(ball_positions, hoop_positions, tolerance=2):
    """
    Determine if a shot was made by checking if the ball's center is within an expanded area around the hoop.
    
    Args:
    - ball_positions (list): List of tuples representing the ball's positions over time.
    - hoop_positions (list): List of tuples representing the hoop's positions over time.
    - tolerance (float): Factor by which to expand the hoop's bounding box dimensions to create a tolerance zone.
    
    Returns:
    - bool: True if a shot is detected, False otherwise.
    """
    global ball_in_hoop

    if len(ball_positions) > 0 and len(hoop_positions) > 0:
        ball_center = ball_positions[-1]
        hoop_x, hoop_y, hoop_width, hoop_height = hoop_positions[-1]
        hoop_center = (int(hoop_x), int(hoop_y))

        # Increase hoop bounding box dimensions by the tolerance factor
        expanded_hoop_width = hoop_width * tolerance
        expanded_hoop_height = hoop_height * tolerance

        hoop_top_left = (hoop_center[0] - expanded_hoop_width // 2, hoop_center[1] - expanded_hoop_height // 2)
        hoop_bottom_right = (hoop_center[0] + expanded_hoop_width // 2, hoop_center[1] + expanded_hoop_height // 2)

        # Check if the ball's center is within the expanded bounding box
        if hoop_top_left[0] < ball_center[0] < hoop_bottom_right[0] and hoop_top_left[1] < ball_center[1] < \
                hoop_bottom_right[1]:
            ball_in_hoop = True
            return True
        else:
            ball_in_hoop = False
    return False


# Function to determine if a goal was made
def detect_goal(ball_positions, hoop_positions):
    if len(ball_positions) > 1 and len(hoop_positions) > 0:
        ball_center = ball_positions[-2]  # Use previous ball position
        hoop_center, hoop_width, hoop_height = hoop_positions[-1]

        # Define points above and below the hoop for accurate goal tracking
        hoop_top = hoop_center[1] - hoop_height // 2
        hoop_bottom = hoop_center[1] + hoop_height // 2

        # Check if the ball's previous position is within the hoop's bounding box
        if (
                hoop_center[0] - hoop_width // 2 < ball_center[0] < hoop_center[0] + hoop_width // 2
                and hoop_top < ball_center[1] < hoop_bottom
        ):
            return True
    return False


def process_frame(frame):
    global shots_made, goals_scored

    # Run object detection inference on the frame
    results = model.infer(frame, tracker="yolo_tracker.yaml")[0]

    # Extract predictions from the results
    predictions = results.predictions

    # Filter out the detections for 'Person'
    filtered_predictions = [pred for pred in predictions if pred.class_name != 'people']

    # Replace the predictions in the results with the filtered predictions
    results.predictions = filtered_predictions

    # Load the results into the supervision Detections API
    detections = sv.Detections.from_inference(results)

    # Annotate the frame with bounding boxes and labels
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Track ball trajectory
    track_ball(predictions, annotated_frame)

    # Track shots and goals
    annotated_frame = track_shots_and_goals(predictions, annotated_frame)

    # # Detect shots and goals
    # if detect_shot(ball_positions, hoop_positions):
    #     shots_made += 1
    #     #cv2.putText(annotated_frame, "Shot Detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     if detect_goal(ball_positions, hoop_positions):
    #         goals_scored += 1
    #         #cv2.putText(annotated_frame, "Goal!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # # Display shot and goal counters
    # cv2.putText(annotated_frame, f"Shots Made: {shots_made}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(annotated_frame, f"Goals Scored: {goals_scored}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotated_frame


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
            # print(f"Ball detected at: {ball_center}")  # Print ball coordinates
            cv2.putText(frame, f"Ball Coordinates: {ball_center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                        2)
            break

    # Track the ball
    if current_ball_position:
        ball_positions.append(current_ball_position)
        prev_ball_position = current_ball_position
    elif prev_ball_position:
        ball_positions.append(prev_ball_position)


def track_shots_and_goals(predictions, frame):
    global shot_counter, goal_counter, cooldown_timer, ball_in_hoop

    # Extract hoop positions from predictions
    hoop_positions = [
        (int(prediction.x), int(prediction.y), int(prediction.width), int(prediction.height))
        for prediction in predictions if prediction.class_name == "hoop"
    ]

    # Track the ball
    track_ball(predictions, frame)

    # Detect shots and goals
    if detect_shot(ball_positions, hoop_positions):
        if cooldown_timer == 0 and ball_in_hoop:
            shot_counter += 1
            if detect_goal(ball_positions, hoop_positions):
                goal_counter += 1
            cooldown_timer = cooldown_frames

    # Update cooldown timer
    if cooldown_timer > 0:
        cooldown_timer -= 1

    # Display shot and goal counts on the frame
    cv2.putText(frame, f"Shots: {shot_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Goals: {goal_counter}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


# def track_hoop(predictions, frame):
#     global hoop_positions
#     current_hoop_position = None

#     for prediction in predictions:
#         if prediction.class_name == hoop_label:
#             x = int(prediction.x)
#             y = int(prediction.y)
#             w = int(prediction.width)
#             h = int(prediction.height)

#             # Calculate center of the hoop
#             center_x = int(x)
#             center_y = int(y)
#             hoop_center = (center_x, center_y)
#             current_hoop_position = (hoop_center, w, h)

#             # Draw a rectangle around the hoop
#             cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
#             cv2.putText(frame, f"Hoop Coordinates: {hoop_center}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#             break

#     if current_hoop_position:
#         hoop_positions.append(current_hoop_position)
#     elif len(hoop_positions) > 0:
#         hoop_positions.append(hoop_positions[-1])


# Example usage
# Process an image
# image_path = "uploads/sample_image.jpg"
# process_image(image_path)

if __name__=='__main__':
    # Process a video
    video_path = "uploads/two_score_two_miss.mp4"
    process_video(video_path)
