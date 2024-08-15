import logging
import sys
import os
from ultralytics import YOLO

from basketball_analytics.basket_ball_class import BasketBallGame
from common.azure_storage import download_blob
from common.utils import load_config

logger = logging.Logger('INFO')

config = load_config('configs/config.ini')
object_detection_model_path = config['paths']['object_detection_model_path']
pose_detection_model_path = config['paths']['pose_detection_model_path']
# Load model objects
object_detection_model = YOLO(object_detection_model_path)
pose_detection_model = YOLO(pose_detection_model_path)
sample_video_path = config['paths']['sample_video_path']
# Define the body part indices and class names
class_names = eval(config['constants']['class_names'])
body_index = eval(config['constants']['body_index'])
azure_connection_string = config['azure']['connection_string']
azure_container_name = config['azure']['container_name']


def analyze_basketball_parameters(video_blob_name):
    try:
        output_blob_name = f"processed_{video_blob_name}"
        shots_data = BasketBallGame(
            object_detection_model,
            pose_detection_model,
            class_names,
            video_blob_name,
            output_blob_name,
            body_index,
            azure_connection_string,
            azure_container_name
        )
        shot_data=shots_data.all_shot_data
        print(f"shot data: {shot_data}")
        return shot_data
    except Exception as e:
        exc_type, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f'Some error with basketball video processing {exc_tb.tb_lineno}th line '
                         f'in {fname}, error {exc_type}')