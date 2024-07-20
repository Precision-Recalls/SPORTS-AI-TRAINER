import datetime
import logging
import os
import random
import sys
import traceback

import cv2
import mediapipe as mp
from ultralytics import YOLO

import common
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

yoga_output_video_path = os.path.abspath(os.path.join(yoga_output_directory, f'yoga_output_video_{now}_{randint}.mp4'))
yoga_output_image_path = os.path.abspath(os.path.join(yoga_output_directory, f'yoga_image_{now}_{randint}.jpg'))

yolo_model = config['paths']['yoga_pose_model']
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
yoga_class = Yoga(yoga_classes, yolo_model, yoga_classifier_model_path, yoga_pose_mapping_filepath,
                  pose_coordinates_path)


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


def landmark_drawer(frame, frame_rate=None):
    output_frame = frame
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(frame)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1))
            output_frame = yoga_class.run(frame, frame_rate)
    return output_frame


def write_frame(writer, frame):
    writer.write(frame)


def analyze_yoga_image(img_path):
    try:
        img = cv2.imread(img_path)
        processed_img = landmark_drawer(img)
        cv2.imwrite(yoga_output_image_path, processed_img)
        logger.info('Output image file got saved!')
    except Exception as e:
        logger.error(f"There is some error in image processing for yoga! :- {e}")


def analyze_yoga_video(video_path, param_list):
    try:
        input_video_cap = cv2.VideoCapture(video_path)
        frame_rate = input_video_cap.get(cv2.CAP_PROP_FPS)
        video_writer = common.utils.video_writer(input_video_cap, yoga_output_video_path)
        while True:
            ret, frame = input_video_cap.read()
            if not ret:
                # End of the video or an error occurred
                break
            processed_frame = landmark_drawer(frame, frame_rate)
            cv2.imshow('output_frame', processed_frame)
            write_frame(video_writer, processed_frame)
            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break
        input_video_cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"Some error with yoga video processing :- {e.__traceback__}")
