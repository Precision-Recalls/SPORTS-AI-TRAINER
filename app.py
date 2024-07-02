from ultralytics import YOLO

from basketball_analytics.basket_ball_class import BasketBallGame
from common.utils import load_config

config = load_config('configs/config.ini')
object_detection_model_path = config['paths']['object_detection_model_path']
pose_detection_model_path = config['paths']['pose_detection_model_path']
# Load model objects
object_detection_model = YOLO(object_detection_model_path)
pose_detection_model = YOLO(pose_detection_model_path)


def get_basketball_configs():
    sample_video_path = config['paths']['sample_video_path']
    # Define the body part indices and class names
    class_names = eval(config['constants']['class_names'])
    body_index = eval(config['constants']['body_index'])
    return sample_video_path, class_names, body_index


def analyze_basketball_video(class_names, video_path, body_index):
    BasketBallGame(object_detection_model, pose_detection_model, class_names, video_path, body_index)


sample_video_path, class_names, body_index = get_basketball_configs()
analyze_basketball_video(class_names, sample_video_path, body_index)
