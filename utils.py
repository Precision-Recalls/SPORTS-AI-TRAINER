import cv2
import numpy as np


def display_angles(frame, angles):
    """
    Display the calculated elbow angles on the frame.
    """
    if "left_elbow" in angles:
        cv2.putText(frame, f"Left Elbow Angle: {angles['left_elbow']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
    if "right_elbow" in angles:
        cv2.putText(frame, f"Right Elbow Angle: {angles['right_elbow']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    """
    ab = np.array([b[0] - a[0], b[1] - a[1]])
    cb = np.array([b[0] - c[0], b[1] - c[1]])
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
