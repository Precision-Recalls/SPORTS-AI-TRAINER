import cv2
import numpy as np
from ultralytics import YOLO

from basketball_analytics.player_class import Player
from basketball_analytics.shot_detector_class import ShotDetector
from common.utils import scale_text, display_angles

model = YOLO(r"C:\Users\Abhay\PycharmProjects\SPORTS-AI-TRAINER\best_yolo8s.pt")
pose_model = YOLO(
    r'C:\Users\Abhay\PycharmProjects\SPORTS-AI-TRAINER\assets\models\yolov8s-pose.pt')  # Assuming you have a YOLO pose model
# Define the body part indices
body_index = {"left_shoulder": 5, "left_elbow": 7, "left_wrist": 9,
              "right_shoulder": 6, "right_elbow": 8, "right_wrist": 10,
              "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}
class_names = ['ball', 'basket', 'person']



def process_frame(frame):
    # Run pose estimation on the frame
    pose_results = pose_model(frame, verbose=False, conf=0.7)[0]
    if pose_results:
        player = Player(frame, pose_results, body_index)
        steps = player.count_steps()
        step_counter += steps # type: ignore
        elbow_angles = player.calculate_elbow_angles()
        display_angles(frame, elbow_angles)

    # Annotate the frame with the step count
    text, position, font_scale, thickness = scale_text(frame, f"Steps: {step_counter}", (10, 30), 1, 2)
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
