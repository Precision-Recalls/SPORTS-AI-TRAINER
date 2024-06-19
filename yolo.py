# from ultralytics import YOLO
# model=YOLO("yolov8m.pt")
# results=model.track(source="/Users/garvagarwal/Desktop/SPORTS-AI-TRAINER/two_score_three_miss.mp4",show=True,tracker='bytetrack.yaml')

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
# Load the pre-trained YOLOv5 model from ultralytics
model = YOLO("yolo_models/yolov8n.pt")


# Open the video file
video_path = "uploads/sample_video.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()