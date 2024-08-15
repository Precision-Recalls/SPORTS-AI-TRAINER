import base64
import os

import cv2

from fitness_analytics.fitness_configs import VIDEO_PATH, WIDTH, HEIGHT, DOWNSIZE_FACTOR, SKIP_FRAME


def get_video(drill_name, video_path=VIDEO_PATH):
    def create_video_path(video_path, drill_name):
        for file in os.listdir(os.path.join(video_path, drill_name)):
            if file.endswith('.mp4'):
                return os.path.join(video_path, drill_name, file)

    video_path = create_video_path(video_path, drill_name)
    print('video path: ', video_path)
    cap, width, height, fps, duration = (None, 0, 0, 0, 0)

    try:
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        episilon = 0.001
        duration = round(frame_count / (fps + episilon))
    except Exception as e:
        print('Could not load video due to: ', e)

    return cap, width, height, fps, duration


def preprocess_image(frame, width=WIDTH, height=HEIGHT, downsize_factor=DOWNSIZE_FACTOR):
    # resize to fixed size
    fixed_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # resize and scale down for faster evaluation
    reduced_frame = cv2.resize(fixed_frame, (int(width / downsize_factor), int(height / downsize_factor)),
                               interpolation=cv2.INTER_AREA)

    # converting to to RGB
    reduced_frame = cv2.cvtColor(reduced_frame, cv2.COLOR_BGR2RGB)

    return fixed_frame, reduced_frame


def return_frame_generator(cap, skip_frame=SKIP_FRAME):
    skip_index = skip_frame
    while cap.isOpened():
        success, image = cap.read()

        if success and skip_index > 0:
            skip_index -= 1
            continue

        elif success:
            skip_index = skip_frame
            fixed_image, reduced_image = preprocess_image(image)
            yield fixed_image, reduced_image

        else:
            break

    cap.release()


def load_video_for_response(video_writer_path):
    with open(video_writer_path, 'rb') as f:
        a = f.read()
    a_byte = base64.b64encode(a)
    a_str = a_byte.decode()
    return a_str
