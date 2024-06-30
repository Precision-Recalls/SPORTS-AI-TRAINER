from ultralytics import YOLO
from basketball_analytics.basket_ball_class import BasketBallGame
from common.utils import load_config

config = load_config('configs/config.ini')
object_detection_model_path = config['paths']['object_detection_model_path']
pose_detection_model_path = config['paths']['pose_detection_model_path']
sample_video_path = config['paths']['sample_video_path']
# Define the body part indices and class names
body_index = config['constants']['body_index']
class_names = config['constants']['class_names']

# Load model objects
object_detection_model = YOLO(object_detection_model_path)
pose_detection_model = YOLO(pose_detection_model_path)


def analyze_basketball_video(video_path):
    BasketBallGame(object_detection_model, pose_detection_model, class_names, video_path, body_index)


def main():
    # If user does not give path of the video then default analysis will be shown
    analyze_basketball_video(sample_video_path)
