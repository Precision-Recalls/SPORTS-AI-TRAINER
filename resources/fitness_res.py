import logging

import mediapipe as mp

from common.azure_service_bus import send_message_to_bus
from common.utils import load_config, DrillType
from fitness_analytics.fitness_class import Fitness

config = load_config('configs/config.ini')
logger = logging.Logger('INFO')

azure_connection_string = config['azure']['connection_string']
azure_input_container_name = config['azure']['input_container_name']
azure_output_container_name = config['azure']['output_container_name']
azure_service_bus_connection_string = config['azure']['azure_service_bus_connection_string']
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def analyze_fitness_video(video_blob_name, drill_name, sender):
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
        send_message_to_bus(sender, fitness_response, DrillType.Fitness.value, video_blob_name)
    except Exception as e:
        logger.error(f"Some error with fitness video processing :- {e}")
