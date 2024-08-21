import datetime
import logging
import random
from azure.servicebus import ServiceBusClient, ServiceBusMessage
import mediapipe as mp

from common.utils import load_config
from fitness_analytics.fitness_class import Fitness

config = load_config('configs/config.ini')
logger = logging.Logger('INFO')

azure_connection_string = config['azure']['connection_string']
azure_input_container_name = config['azure']['input_container_name']
azure_output_container_name = config['azure']['output_container_name']
azure_service_bus_connection_string=config['azure']['azure_service_bus_connection_string']
servicebus_client = ServiceBusClient.from_connection_string(azure_service_bus_connection_string)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
queue_client = servicebus_client.get_queue_client("your_queue_name")
sender = queue_client.get_sender()

def analyze_fitness_video(video_blob_name, drill_name):
    try:
        output_blob_name = f"processed_{video_blob_name}"
        fitness_response = Fitness(
            drill_name,
            video_blob_name,
            output_blob_name,
            azure_connection_string,
            azure_input_container_name, azure_output_container_name
        )
        logger.info(f"Fitness video's final stats are as follows :- {fitness_response.response}")
        fitness_response = fitness_response.response
        # Send a message to the Service Bus topic
        message = ServiceBusMessage(str(fitness_response))
        message.application_properties = {"MessageType": "Fitness","VideoID":video_blob_name}
        sender.send_message(message)
    except Exception as e:
        logger.error(f"Some error with fitness video processing :- {e}")
