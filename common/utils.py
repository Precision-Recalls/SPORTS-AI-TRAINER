import cv2
import numpy as np


def scale_text(frame, text, position, font_scale, thickness):
    """Scale text size and position according to frame size."""
    frame_height, frame_width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_scale * min(frame_width, frame_height) / 1000  # Scale based on frame size
    thickness = int(thickness * min(frame_width, frame_height) / 1000)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    # position = (int(position[0] * frame_width), int(position[1] * frame_height))
    return text, position, font_scale, thickness


def display_angles(frame, angles):
    """
    Display the calculated elbow angles on the frame.
    """
    if "left_elbow" in angles:
        text, position, font_scale, thickness = scale_text(frame, f"Left Elbow Angle: {angles['left_elbow']:.2f}",
                                                           (10, 60), 1, 2)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    if "right_elbow" in angles:
        text, position, font_scale, thickness = scale_text(frame, f"Right Elbow Angle: {angles['right_elbow']:.2f}",
                                                           (10, 90), 1, 2)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


def calculate_angle(a, b, c):
    try:
        """
        Calculate the angle between three points.
        """
        ab = np.array([b[0] - a[0], b[1] - a[1]])
        cb = np.array([b[0] - c[0], b[1] - c[1]])
        cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except Exception as e:
        print(f"There is some issue with angle calculation:- {e}")
