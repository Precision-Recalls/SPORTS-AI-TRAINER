import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolo_models/yolov8s.pt")


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    cap.release()
    return frames, frame_count


def detect_players_and_ball(frames):
    results = []
    for frame in frames:
        result = model(frame)
        results.append(result)
    return results


# Usage
video_path = 'test/one_score_one_miss.mp4'
frames, total_frames = extract_frames(video_path)

# Usage
detections = detect_players_and_ball(frames)

print(f"Extracted {total_frames} frames")
