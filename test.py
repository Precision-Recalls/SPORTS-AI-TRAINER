from inference import get_model
import supervision as sv
import cv2
from ultralytics import YOLO
from inference import get_model

#model = get_model(model_id="basketball-detection-dn6fg/4")

#results = model.infer("https://media.roboflow.com/inference/people-walking.jpg")

# Initialize the model
model=YOLO("yolov8n.pt")

# Create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def process_frame(frame):
    # Run inference on the frame
    results = model(frame)[0]
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    # Load the results into the supervision Detections API
    #detections = sv.Detections.from_inference(results)

    # Annotate the frame with bounding boxes and labels
    #annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    #annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

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


# Example usage
# Process an image
# image_path = "test/sample_image.jpg"
# process_image(image_path)

# Process a video
video_path = "/Users/garvagarwal/Desktop/SPORTS-AI-TRAINER/test/The player with no arc on their shot 😂🏀 #shorts.mp4"
process_video(video_path)
