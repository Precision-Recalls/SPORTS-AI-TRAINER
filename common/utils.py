import cv2
import numpy as np
import configparser
from azure.storage.blob import BlobClient
import os
def load_config(config_file):
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
    except Exception as e:
        print(f"Error occurred in configuration loading: {e}")


def scale_text(frame, text, position, font_scale, thickness):
    """Scale text size and position according to frame size."""
    frame_height, frame_width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_scale * min(frame_width, frame_height) / 1000  # Scale based on frame size
    thickness = int(thickness * min(frame_width, frame_height) / 1000)
    return text, position, font_scale, thickness


def add_text(image_text_dict, img):
    for _, values in image_text_dict.items():
        img_text, text_pos = values['text'], values['position']
        text, position, font_scale, thickness = scale_text(img, img_text, text_pos, 1, 2)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    return img


def display_angles(frame, angles):
    """
    Display the calculated elbow angles on the frame.
    """
    if "left_elbow" in angles and angles["left_elbow"] is not None:
        text, position, font_scale, thickness = scale_text(frame, f"Left Elbow Angle: {angles['left_elbow']:.2f}",
                                                           (10, 115), 1, 2)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    if "right_elbow" in angles and angles["right_elbow"] is not None:
        text, position, font_scale, thickness = scale_text(frame, f"Right Elbow Angle: {angles['right_elbow']:.2f}",
                                                           (10, 135), 1, 2)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


def calculate_angle(a, b, c):
    try:
        """
        Calculate the angle between three points.
        """
        ab = np.array([b[0] - a[0], b[1] - a[1]])
        cb = np.array([b[0] - c[0], b[1] - c[1]])
        if np.linalg.norm(ab) == 0 or np.linalg.norm(cb) == 0:
            # Return the last calculated angle if available, otherwise return None
            if hasattr(calculate_angle, 'last_angle'):
                return calculate_angle.last_angle
            else:
                return None

        cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip cosine_angle to [-1, 1] range
        calculate_angle.last_angle = np.degrees(angle)  # Store the last calculated angle
        return calculate_angle.last_angle
    except Exception as e:
        print(f"There is some issue with angle calculation:- {e}")


# def video_writer(cap, output_path):
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
def video_writer(cap, blob_client):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    class BlobVideoWriter:
        def __init__(self, blob_client, fourcc, fps, frameSize):
            self.blob_client = blob_client
            self.writer = cv2.VideoWriter('temp.mp4', fourcc, fps, frameSize)
        
        def write(self, frame):
            self.writer.write(frame)
        
        def release(self):
            self.writer.release()
            with open('temp.mp4', 'rb') as video_file:
                self.blob_client.upload_blob(video_file.read(), overwrite=True)
            os.remove('temp.mp4')
    
    return BlobVideoWriter(blob_client, fourcc, fps, (frame_width, frame_height))