import cv2
import numpy as np
import supervision as sv
from inference import get_model
from ultralytics import YOLO

# Initialize the object detection and pose estimation models
from basketball_analytics.ball_analytics import track_ball
from basketball_analytics.player_analytics import draw_pose_landmarks_and_bboxes, calculate_elbow_angles, count_steps
from utils import display_angles

model = get_model(model_id="tracer-basketball/3")
pose_model = YOLO("/Users/garvagarwal/Desktop/SPORTS-AI-TRAINER/assets/models/yolov8s-pose.pt")  # Assuming you have a YOLO pose model

# Create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

step_counter = 0
def process_frame(frame):
    global step_counter, prev_left_ankle_y, prev_right_ankle_y, wait_frames

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

    # Run pose estimation on the frame
    pose_results = pose_model(frame, verbose=False, conf=0.4)[0]

    # Draw landmarks and bounding boxes for pose estimation
    annotated_frame = draw_pose_landmarks_and_bboxes(annotated_frame, pose_results)
    # Count steps
    steps = count_steps(pose_results)
    step_counter += steps

    # Track ball trajectory
    track_ball(predictions, annotated_frame)

    # Calculate and display elbow angles
    elbow_angles = calculate_elbow_angles(pose_results)
    display_angles(annotated_frame, elbow_angles)

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


if __name__ == '__main__':
    # image_path = "test/sample_image.jpg"
    # process_image(image_path)

    # Process a video
    video_path = "/Users/garvagarwal/Desktop/SPORTS-AI-TRAINER/test/two_score_two_miss.mp4"
    process_video(video_path)
