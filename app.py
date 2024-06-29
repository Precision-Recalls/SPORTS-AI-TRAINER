from ultralytics import YOLO
from basketball_analytics.shot_detector_class import ShotDetector

model = YOLO(r"assets\models\best_yolo8s.pt")
pose_model = YOLO(r'assets\models\yolov8s-pose.pt')
# Define the body part indices
body_index = {"left_shoulder": 5, "left_elbow": 7, "left_wrist": 9,
              "right_shoulder": 6, "right_elbow": 8, "right_wrist": 10,
              "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}
class_names = ['ball', 'basket', 'person']


def analyze_basketball_video(video_path='test\two_score_two_miss.mp4'):
    shot_detector = ShotDetector(model, pose_model, class_names, video_path, body_index)
