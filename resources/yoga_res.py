import datetime
import logging
import random

import mediapipe as mp
from ultralytics import YOLO

from common.azure_storage import download_blob
from common.utils import load_config
from yoga_analytics.yoga_class import Yoga
from yoga_analytics.yoga_classifier_trainer import YogaClassifierTrainingClass

config = load_config('configs/config.ini')
logger = logging.Logger('INFO')

now = datetime.datetime.now().date()
randint = random.randint(100, 200)

yoga_classifier_model_path = config['paths']['yoga_classifier_model_path']
pose_coordinates_path = config['paths']['pose_coordinates_path']
yoga_output_directory = config['paths']['yoga_output_path']
image_folder = config['paths']['yoga_poses_image_folder']
yoga_pose_mapping_filepath = config['paths']['yoga_pose_mapping_filepath']
yoga_classes = eval(config['constants']['yoga_classes'])
azure_connection_string = config['azure']['connection_string']
azure_container_name = config['azure']['container_name']


yolo_model = YOLO("assets/models/yolov8m-pose.pt")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def start_yoga_classifier_training():
    try:
        start_time = datetime.datetime.now()
        logger.info(f"Training started at :- {start_time}")
        yoga_classification_trainer = YogaClassifierTrainingClass(image_folder, yoga_classifier_model_path, yolo_model,
                                                                  pose_coordinates_path)
        yoga_classification_trainer.run()
        end_time = datetime.datetime.now()
        logger.info(f"Training ended at :- {end_time} and it took :- {end_time - start_time}")
    except Exception as e:
        logger.error(f"There is some issue with yoga classifier training :- {e}")


def analyze_yoga_video(video_blob_name, param_list):
    try:
        output_blob_name = f"processed_{video_blob_name}"
        yoga_class = Yoga(yoga_classes, video_blob_name, output_blob_name, yolo_model, yoga_classifier_model_path,
                          yoga_pose_mapping_filepath, pose_coordinates_path, azure_connection_string,
                          azure_container_name)
        yoga_class_response = yoga_class.yoga_final_stats
        yoga_final_response_data = [{key: yoga_data[key] for key in param_list} for yoga_data in
                                    yoga_class_response.to_list()]
        video_data = download_blob(output_blob_name)
        return yoga_final_response_data
    except Exception as e:
        logger.error(f"Some error with yoga video processing :- {e}")
