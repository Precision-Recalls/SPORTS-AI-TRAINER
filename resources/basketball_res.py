import logging
import sys
import os
from ultralytics import YOLO
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from basketball_analytics.basket_ball_class import BasketBallGame
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
azure_input_container_name = config['azure']['input_container_name']
azure_output_container_name=config['azure']['output_container_name']
azure_service_bus_connection_string=config['azure']['azure_service_bus_connection_string']
servicebus_client = ServiceBusClient.from_connection_string(azure_service_bus_connection_string)
queue_client = servicebus_client.get_queue_client("your_queue_name")
sender = queue_client.get_sender()

def analyze_basketball_parameters(video_blob_name):
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
            azure_input_container_name,azure_output_container_name
        )
        shots_data = basketball_cls.all_shot_data
        # Send a message to the Service Bus topic
        message = ServiceBusMessage(str(shots_data))
        message.application_properties = {"MessageType": "Basketball","VideoID":video_blob_name}
        sender.send_message(message)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f'Some error with basketball video processing {exc_tb.tb_lineno}th line '
                         f'in {fname}, error {exc_type}')