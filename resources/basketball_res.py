import logging
import os
import sys

from ultralytics import YOLO

from basketball_analytics.basket_ball_class import BasketBallGame
from common.azure_service_bus import send_message_to_bus
from common.utils import load_config, DrillType

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
azure_input_container_name = config['azure']['input_container_name']
azure_output_container_name = config['azure']['output_container_name']


def analyze_basketball_parameters(video_blob_name, sender):
    try:
        output_blob_name = f"processed_{video_blob_name}"
        basketball_cls = BasketBallGame(
            object_detection_model,
            pose_detection_model,
            class_names,
            video_blob_name,
            output_blob_name,
            body_index,
            azure_connection_string,
            azure_input_container_name, azure_output_container_name
        )
        shots_data = basketball_cls.all_shot_data
        send_message_to_bus(sender, shots_data, DrillType.BasketBall.value, video_blob_name)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f'Some error with basketball video processing {exc_tb.tb_lineno}th line '
                     f'in {fname}, error {exc_type}')
