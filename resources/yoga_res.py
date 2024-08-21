import datetime
import logging
import random
import sys
import os
import mediapipe as mp
from ultralytics import YOLO

from common.utils import load_config
from yoga_analytics.yoga_class import Yoga
from yoga_analytics.yoga_classifier_trainer import YogaClassifierTrainingClass

config = load_config('configs/config.ini')
logger = logging.Logger('INFO')

now = datetime.datetime.now().date()
randint = random.randint(100, 200)

yoga_classifier_model_path = config['paths']['yoga_classifier_model_path']
yoga_yolo_model_path = config['paths']['yoga_pose_model']
pose_coordinates_path = config['paths']['pose_coordinates_path']
image_folder = config['paths']['yoga_poses_image_folder']
yoga_pose_mapping_filepath = config['paths']['yoga_pose_mapping_filepath']
yoga_classes = eval(config['constants']['yoga_classes'])
azure_connection_string = config['azure']['connection_string']
azure_input_container_name = config['azure']['input_container_name']
azure_output_container_name = config['azure']['output_container_name']

yolo_model = YOLO(yoga_yolo_model_path)

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
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f'There is some issue with yoga classifier training {exc_tb.tb_lineno}th line '
                     f'in {fname}, error {exc_type}')


def analyze_yoga_video(video_blob_name):
    try:
        output_blob_name = f"processed_{video_blob_name}"
        yoga_class = Yoga(yoga_classes, video_blob_name, output_blob_name, yolo_model, yoga_classifier_model_path,
                          yoga_pose_mapping_filepath, pose_coordinates_path, azure_connection_string,
                          azure_input_container_name, azure_output_container_name)
        yoga_class_response = yoga_class.yoga_final_stats
        return yoga_class_response
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f'Some error with yoga video processing {exc_tb.tb_lineno}th line '
                     f'in {fname}, error {exc_type}')
