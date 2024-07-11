import datetime
import logging
import os
import random

import cv2
import mediapipe as mp
from flask import Flask, render_template
from ultralytics import YOLO

import common
from common.utils import load_config
from yoga_analytics.yoga_class import Yoga
from yoga_analytics.yoga_classifier_trainer import YogaClassifierTrainingClass

app = Flask(__name__)

now = datetime.datetime.now().date()
randint = random.randint(100, 200)

logger = logging.Logger('INFO')

config = load_config('../configs/config.ini')

yoga_classifier_model_path = config['paths']['yoga_classifier_model_path']
pose_coordinates_path = config['paths']['pose_coordinates_path']
yoga_output_directory = config['paths']['yoga_output_path']
image_folder = config['paths']['yoga_poses_image_folder']
yoga_pose_mapping_filepath = config['paths']['yoga_pose_mapping_filepath']
yoga_classes = eval(config['constants']['yoga_classes'])

yoga_output_video_path = os.path.abspath(os.path.join(yoga_output_directory, f'yoga_output_video_{now}_{randint}.mp4'))
yoga_output_image_path = os.path.abspath(os.path.join(yoga_output_directory, f'yoga_image_{now}_{randint}.jpg'))

yolo_model = YOLO("yolov8m-pose.pt")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# yoga_classification_trainer = YogaClassifierTrainingClass(image_folder, yoga_classifier_model_path, yolo_model,
#                                                           pose_coordinates_path)
# yoga_classification_trainer.run()

yoga_class = Yoga(yoga_classes, yolo_model, yoga_classifier_model_path, yoga_pose_mapping_filepath,
                  pose_coordinates_path)


def process_image(img_path):
    try:
        img = cv2.imread(img_path)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(img)
            # Render detections
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1))
            output_image = yoga_class.run(img)
            cv2.imwrite(yoga_output_image_path, output_image)
            logger.info('Output image file got saved!')
    except Exception as e:
        logger.error(f"There is some error in image processing for yoga! :- {e}")


def process_video(video_path, output_video_path):
    input_video_cap = cv2.VideoCapture(video_path)
    video_writer = common.utils.video_writer(input_video_cap, output_video_path)
    while True:
        ret, frame = input_video_cap.read()
        if not ret:
            # End of the video or an error occurred
            break
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(frame)
            # Render detections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1))
            output_frame = yoga_class.run(frame)
            video_writer.write(output_frame)
    input_video_cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


# # @app.route('/prediction', methods=['POST'])
# def predict(image):
#     # Get the image file from the request
#     image_file = request.files['file']
#     # Save the image file to a temporary location
#     image_path = 'temp.png'
#     image_file.save(image_path)
#
#     # Load the model
#     model = load_yoga_classifier_model()
#
#     # Make prediction
#     prediction = make_prediction(model, image_path)
#
#     # Remove the temporary image file
#     os.remove(image_path)
#     print(f"Image prediction is as follows:- {prediction}")


if __name__ == '__main__':

    # ------------------- Make a prediction -----------
    image_path = r'../yoga_analytics/yoga_pose_data_archive/Virabhadrasana Two/File7.png'
    process_image(image_path)
    # video_path = ''
    # process_video(video_path)
    # Close if 'q' is clicked
    if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
        exit()
