import base64
import logging
import os
from ultralytics import YOLO
from basketball_analytics.basket_ball_class import BasketBallGame
from common.utils import load_config
from common.azure_storage import upload_blob,download_blob
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

output_folder = config['paths']['output_folder']

def analyze_basketball_parameters(video_blob_name, socketio, param_list):
    # ...
    output_blob_name = f"processed_{video_blob_name}"
    # filename, file_extension = os.path.splitext(os.path.basename(file_path))
    # processed_filename = f"processed_{filename}{file_extension}"
    # processed_video_path = os.path.join(output_folder, processed_filename)

    shots_data = BasketBallGame(
        object_detection_model,
        pose_detection_model,
        class_names,video_blob_name,
        output_blob_name,
        # file_path,
        # processed_video_path,
        body_index
    )
    # ...
    shots_response_data = [{key: shot_data[key] for key in param_list} for shot_data in shots_data.to_list()]
    video_data = download_blob(output_blob_name)
    encoded_video = base64.b64encode(video_data).decode('utf-8')
    socketio.emit('video_processed', {'video': encoded_video})
    socketio.emit('video_processed', {'analytics': shots_response_data})
    # with open(processed_video_path, 'rb') as video_file:
    #     video_data = video_file.read()
    #     encoded_video = base64.b64encode(video_data).decode('utf-8')

    # # Emit the complete video to the client
    # socketio.emit('video_processed', {'video': encoded_video})
    # socketio.emit('video_processed', {'analytics': shots_response_data})
