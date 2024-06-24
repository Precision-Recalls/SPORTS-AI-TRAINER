from inference import get_model
import supervision as sv
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import tempfile


# Initialize the object detection and pose estimation models
model = get_model(model_id="tracer-basketball/3")
pose_model = YOLO("AI-Basketball-Referee/yolov8s-pose.pt") # Assuming you have a YOLO pose model

# Create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


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

# Initialize ball trajectory
ball_positions = []
prev_ball_position = None
ball_label = "basketball"  # Adjust this label based on your model's output


def process_frame(frame):
    global step_counter, prev_left_ankle_y, prev_right_ankle_y, wait_frames

    # Run object detection inference on the frame
    results = model.infer(frame,tracker="bytetrack.yaml")[0]

     # Extract predictions from the results
    predictions = results.predictions

    # Filter out the detections for 'Person'
    filtered_predictions = [pred for pred in predictions if pred.class_name != 'people']

    # Replace the predictions in the results with the filtered predictions
    results.predictions = filtered_predictions

    print(predictions)
    # Run pose estimation on the frame
    pose_results =pose_model(frame,verbose=False,conf=0.4)[0]

    # Load the results into the supervision Detections API
    detections = sv.Detections.from_inference(results)

    # Annotate the frame with bounding boxes and labels
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    
    # Count steps
    steps = count_steps(pose_results)
    step_counter += steps

    # Track ball trajectory
    track_ball(predictions, annotated_frame)

    # Calculate and display elbow angles
    elbow_angles = calculate_elbow_angles(pose_results)
    display_elbow_angles(annotated_frame, elbow_angles)

    # Annotate the frame with the step count
    cv2.putText(annotated_frame, f"Steps: {step_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
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
            ball_center = (center_x,center_y)

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
            cv2.putText(frame, f"Ball Coordinates: {ball_center}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
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

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    """
    ab = np.array([b[0] - a[0], b[1] - a[1]])
    cb = np.array([b[0] - c[0], b[1] - c[1]])
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

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

def display_elbow_angles(frame, angles):
    """
    Display the calculated elbow angles on the frame.
    """
    if "left_elbow" in angles:
        cv2.putText(frame, f"Left Elbow Angle: {angles['left_elbow']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if "right_elbow" in angles:
        cv2.putText(frame, f"Right Elbow Angle: {angles['right_elbow']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Example usage
# Process an image
#image_path = "uploads/sample_image.jpg"
#process_image(image_path)

# Process a video
video_path = "uploads/two_score_two_miss.mp4"
process_video(video_path)


