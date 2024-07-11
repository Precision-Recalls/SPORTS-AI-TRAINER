import base64
import logging

from ultralytics import YOLO

from app import config, socketio
from basketball_analytics.basket_ball_class import BasketBallGame

logger = logging.Logger('INFO')

object_detection_model_path = config['paths']['object_detection_model_path']
pose_detection_model_path = config['paths']['pose_detection_model_path']
# Load model objects
object_detection_model = YOLO(object_detection_model_path)
pose_detection_model = YOLO(pose_detection_model_path)
sample_video_path = config['paths']['sample_video_path']
basketball_output_video_path = config['paths']['basketball_output_video_path']
# Define the body part indices and class names
class_names = eval(config['constants']['class_names'])
body_index = eval(config['constants']['body_index'])

output_folder = config['paths']['output_folder']

allowed_extensions = eval(config['constants']['allowed_extensions'])


def analyze_basketball_parameters(file_path, param_list):
    shots_data = BasketBallGame(
        object_detection_model,
        pose_detection_model,
        class_names,
        file_path,
        basketball_output_video_path,
        body_index
    )
    shots_response_data = [{key: shot_data[key] for key in param_list} for shot_data in shots_data.to_list()]
    with open(basketball_output_video_path, 'rb') as video_file:
        video_data = video_file.read()
        encoded_video = base64.b64encode(video_data).decode('utf-8')

    # Emit the complete video to the client
    socketio.emit('video_processed', {'video': encoded_video})
    socketio.emit('video_processed', {'analytics': shots_response_data})
