import logging
import pickle

import numpy as np
import torch

from common.utils import add_text
from yoga_analytics.yoga_classifier_trainer import YogaClassifier

logger = logging.Logger('CRITICAL')


def create_pose_mappings(yoga_pose_mapping_filepath):
    pose_map = {}
    with open(yoga_pose_mapping_filepath) as pose_map_file:
        file_contents = eval(pose_map_file.read())['Poses']
        for pose in file_contents:
            pose_map[pose['sanskrit_name']] = pose['english_name']
    return pose_map


# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Function to calculate PCK accuracy
def calculate_pck(detected_keypoints, ground_truth_keypoints, threshold):
    correct_keypoints = 0
    total_keypoints = len(ground_truth_keypoints)

    for detected, ground_truth in zip(detected_keypoints, ground_truth_keypoints):
        distance = euclidean_distance(detected, ground_truth)
        if distance <= threshold:
            correct_keypoints += 1
    return correct_keypoints / total_keypoints


class Yoga:
    def __init__(self, yoga_classes, yolo_model, yoga_classifier_model_path, yoga_pose_mapping_filepath,
                 pose_coordinates_path):
        self.yoga_classes = yoga_classes
        self.yoga_classifier_model_path = yoga_classifier_model_path
        self.model_yolo = yolo_model
        self.pose_map = create_pose_mappings(yoga_pose_mapping_filepath)
        self.pose_coordinates_path = pose_coordinates_path
        self.clf_model = None
        self.image = None
        self.repetition_count = 0
        self.pck_accuracy = 0.0
        self.prev_prediction = 'No Pose'
        self.current_prediction = 'No Pose'
        self.pose_counter = {}
        self.predicted_keypoints = None

    def run(self, image):
        self.clf_model = self.load_yoga_classifier_model()
        self.image = image
        self.make_prediction()
        self.count_repetition()
        self.calculate_pose_accuracy()
        output_image = self.display_parameters()
        return output_image

    def load_yoga_classifier_model(self):
        try:
            model_pose = YogaClassifier(num_classes=len(self.yoga_classes), input_length=32)
            model_pose.load_state_dict(
                torch.load(self.yoga_classifier_model_path))
            model_pose.eval()
            logger.info('Yoga classifier model got loaded!')
            return model_pose
        except Exception as e:
            logger.error(f'There is some error in classifier model loading :- {e}')

    def get_pose_keypoints(self, image):
        results = self.model_yolo(image, verbose=False)
        for r in results:
            keypoints = r.keypoints.xyn.cpu().numpy()[0]
            keypoints = keypoints.reshape((1, keypoints.shape[0] * keypoints.shape[1]))[0].tolist()
            return keypoints

    def make_prediction(self):
        try:
            self.predicted_keypoints = self.get_pose_keypoints(self.image)
            # Preprocess keypoints data
            if self.predicted_keypoints:
                keypoints_tensor = torch.tensor(self.predicted_keypoints[2:], dtype=torch.float32).unsqueeze(0)
                self.clf_model.cpu()
                self.clf_model.eval()
                with torch.no_grad():
                    logit = self.clf_model(keypoints_tensor)
                    pred = torch.softmax(logit, dim=1).argmax(dim=1).item()
                    self.current_prediction = self.yoga_classes[pred]
                logger.info(f"Prediction for the current frame is :- {self.current_prediction}")
        except Exception as e:
            logger.error(f"Some issue with prediction method :- {e}")

    def count_repetition(self):
        if self.prev_prediction != self.current_prediction:
            if self.current_prediction not in self.pose_counter:
                self.pose_counter[self.current_prediction] = 1
            self.pose_counter[self.current_prediction] += 1

    def calculate_pose_accuracy(self):
        # Calculate the PCK accuracy
        threshold = 0.7  # Example threshold distance
        with open(self.pose_coordinates_path, "rb") as fp:  # Unpickling
            pose_coordinates = pickle.load(fp)
        if self.current_prediction != 'No Pose':
            ground_truth_keypoints = pose_coordinates[self.current_prediction]
            self.pck_accuracy = round(calculate_pck(self.predicted_keypoints, ground_truth_keypoints, threshold), 2)
            logger.info(f"Pose accuracy is :- {self.pck_accuracy}")

    def display_parameters(self):
        # Display description of the last shot made
        final_prediction = 'No Pose'
        image_text_dict = {
            'pose': {'text': f"Pose: {final_prediction}", 'position': (10, 20)}
        }
        if self.current_prediction != 'No Pose':
            final_prediction = self.pose_map[self.current_prediction]
            image_text_dict = {
                'pose': {'text': f"Pose: {final_prediction}", 'position': (10, 20)},
                'accuracy': {'text': f"Pose Accuracy: {self.pck_accuracy}", 'position': (10, 40)},
                'count': {'text': f"Count: {self.pose_counter[self.current_prediction]}", 'position': (10, 60)}
            }
        return add_text(image_text_dict, self.image)
