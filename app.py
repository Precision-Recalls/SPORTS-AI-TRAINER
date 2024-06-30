from ultralytics import YOLO
from basketball_analytics.basket_ball_class import BasketBallGame
from common.utils import load_config

config = load_config('configs/config.ini')
object_detection_model_path = config['paths']['object_detection_model_path']
pose_detection_model_path = config['paths']['pose_detection_model_path']
sample_video_path = config['paths']['sample_video_path']
# Define the body part indices and class names
class_names = config['constants']['class_names']
body_index = {"left_shoulder": 5, "left_elbow": 7, "left_wrist": 9,
              "right_shoulder": 6, "right_elbow": 8, "right_wrist": 10,
              "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}
# Load model objects
object_detection_model = YOLO(object_detection_model_path)
pose_detection_model = YOLO(pose_detection_model_path)


def analyze_basketball_video(video_path):
    BasketBallGame(object_detection_model, pose_detection_model, class_names, video_path, body_index)


analyze_basketball_video(sample_video_path)
