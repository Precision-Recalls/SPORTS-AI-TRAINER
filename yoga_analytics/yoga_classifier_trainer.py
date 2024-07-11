import logging
import os
import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, F1Score
from tqdm.auto import tqdm

logger = logging.Logger('CRITICAL')

image_file_list = []


def get_all_image_file_list(image_folder):
    for label in os.listdir(image_folder):
        label_dir = os.path.join(image_folder, label)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            image_file_list.append((label, img_path))
    logger.info(f"Total of {len(image_file_list)} image files are there for training")
    return image_file_list


class YogaClassifier(nn.Module):
    def __init__(self, num_classes, input_length):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input_length, out_features=64)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(in_features=64, out_features=64)
        self.outlayer = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.outlayer(x)
        return x


class YogaClassifierTrainingClass:
    def __init__(self, image_folder, model_output_path, yolo_model, pose_coordinates_path):
        self.image_folder = image_folder
        self.model_yolo = yolo_model
        self.model_output_path = model_output_path
        self.pose_coordinates_path = pose_coordinates_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 32
        self.input_length = 0
        self.num_classes = 0
        self.X = []
        self.y = []
        self.train_dataloader = None
        self.test_dataloader = None
        self.model = None
        self.run()

    def run(self):
        get_all_image_file_list(self.image_folder)
        self.prepare_training_data()
        logger.info('Data got prepared!')
        self.split_preprocess_data()
        logger.info('Data got splitted!')
        self.training()
        logger.info('Training done!')
        self.save_pose_coordinates()
        logger.info('Ground reference pose coordinates saved!')

    def save_pose_coordinates(self):
        pose_coordinates = {}
        for label, img_path in image_file_list:
            keypoints = self.get_pose_keypoints(img_path)
            pose_coordinates.update({label: keypoints})
            break
        with open(self.pose_coordinates_path, "wb") as fp:  # Pickling
            pickle.dump(pose_coordinates, fp)
        logger.info("Pose coordinates saved successfully!")

    def get_pose_keypoints(self, image_path):
        results = self.model_yolo.predict(image_path, verbose=False)
        for r in results:
            keypoints = r.keypoints.xyn.cpu().numpy()[0]
            keypoints = keypoints.reshape((1, keypoints.shape[0] * keypoints.shape[1]))[0].tolist()
            return keypoints

    def prepare_training_data(self):
        data = []
        for label, img_path in image_file_list:
            keypoints = self.get_pose_keypoints(img_path)
            keypoints.append(img_path)  # insert image path
            keypoints.append(label)  # insert image label
            data.append(keypoints)

        total_features = len(data[0])
        df = pd.DataFrame(
            data=data, columns=[f"x{i}" for i in range(total_features)]) \
            .rename({"x34": "image_path", "x35": "label"}, axis=1)

        df = df.dropna()  # delete undetected pose
        df = df.iloc[:, 2:]

        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
        self.num_classes = len(le.classes_)

        self.X = df.drop(["label", "image_path"], axis=1).values
        self.input_length = self.X.shape[1]

        self.y = df['label'].values

    def split_preprocess_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1)

        X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
        y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

        train_tensor = TensorDataset(X_train, y_train)
        test_tensor = TensorDataset(X_test, y_test)

        self.train_dataloader = DataLoader(dataset=train_tensor, batch_size=self.BATCH_SIZE, shuffle=True)
        self.test_dataloader = DataLoader(dataset=test_tensor, batch_size=self.BATCH_SIZE, shuffle=False)

    def calculate_training_metric(self, is_train=False):
        accuracy_score = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        f1_score = F1Score(task="multiclass", num_classes=self.num_classes).to(self.device)
        if is_train:
            x, y = self.train_dataloader.dataset.x, self.train_dataloader.dataset.y
        else:
            x, y = self.test_dataloader.dataset.x, self.test_dataloader.dataset.y
        y_pred = self.model(x)
        accuracy = accuracy_score(y_pred, y)
        f1_score = f1_score(y_pred, y)
        return accuracy, f1_score

    def training(self):

        self.model = YogaClassifier(num_classes=self.num_classes, input_length=self.input_length).to(self.device)
        optimizer = torch.optim.Adam(lr=0.001, params=self.model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        epochs = 200

        for _ in tqdm(range(epochs)):
            self.model.train()
            for batch, (X, y) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()

        # training data
        training_accuracy, training_f1_score = self.calculate_training_metric(is_train=True)
        logger.info(
            f'Classifier accuracy on training set is :- {training_accuracy} & f1-score is :- {training_f1_score}')

        # validation data
        test_accuracy, test_f1_score = self.calculate_training_metric()
        logger.info(f'Classifier accuracy on test set is :- {test_accuracy} & f1-score is :- {test_f1_score}')

        torch.save(self.model.state_dict(), self.model_output_path)
        logger.info('classifier model got saved')
