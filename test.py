import cv2
from ultralytics import YOLO

# Initialize the model
model=YOLO("/Users/garvagarwal/Desktop/SPORTS-AI-TRAINER/models/best_yolo8m.pt")


def process_frame(frame):
    
    results=model(frame,conf=0.8)
    print(results)
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
   
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen
        #result.save(filename="result.jpg")  # save to disk
    # # Annotate the frame with bounding boxes and labels
    # annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

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
    cv2.waitKey(1)
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
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release video capture and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


# Example usage
# Process an image
# image_path = "test/sample_image.jpg"
# process_image(image_path)

# Process a video
video_path = "/Users/garvagarwal/Desktop/SPORTS-AI-TRAINER/test/one_score_one_miss.mp4"
process_video(video_path)
