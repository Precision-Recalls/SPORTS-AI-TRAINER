from roboflow import Roboflow
import supervision as sv
import cv2

# Initialize Roboflow model
rf = Roboflow(api_key="StCCjJojqc2pNDrsqvKB")
project = rf.workspace().project("tracer-basketball")
model = project.version(3).model

# Open the video file
video_path = "uploads/multi_angle.mp4"  # Change this to your video file path
video_capture = cv2.VideoCapture(video_path)

# Initialize annotators
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoundingBoxAnnotator()


while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (640, 360))
    # Predict and get results
    result = model.predict(resized_frame, confidence=20, overlap=40).json()
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_inference(result)

    # Annotate the frame
    annotated_frame = bounding_box_annotator.annotate(scene=resized_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Display the frame
    cv2.imshow('Annotated Video', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()


