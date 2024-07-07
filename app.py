import base64
import os
from threading import Thread

import cv2
import numpy as np
from flask import Flask, send_file, Response, request, jsonify
from flask_socketio import SocketIO
from ultralytics import YOLO

from basketball_analytics.basket_ball_class import BasketBallGame
from common.utils import load_config

app = Flask(__name__)

config = load_config('configs/config.ini')
object_detection_model_path = config['paths']['object_detection_model_path']
pose_detection_model_path = config['paths']['pose_detection_model_path']
# Load model objects
object_detection_model = YOLO(object_detection_model_path)
pose_detection_model = YOLO(pose_detection_model_path)
sample_video_path = config['paths']['sample_video_path']
basketball_output_video_path = config['paths']['basketball_output_video_path']
# Define the body part indices and class names
class_names = eval(config['constants']['class_names'])
body_index = eval(config['constants']['body_index'])

output_folder = config['paths']['output_folder']
upload_folder = config['paths']['upload_folder']
# Directory to save frames
frame_upload_folder = config['paths']['frame_upload_folder']

allowed_extensions = eval(config['constants']['allowed_extensions'])
# Ensure the upload folder exists
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(frame_upload_folder, exist_ok=True)
# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def analyze_parameters(file_path, param_list):
    shots_data = BasketBallGame(
        object_detection_model,
        pose_detection_model,
        class_names,
        file_path,
        basketball_output_video_path,
        body_index
    )
    shots_response_data = [{key: shot_data[key] for key in param_list} for shot_data in shots_data.to_list()]
    with open(basketball_output_video_path, 'rb') as video_file:
        video_data = video_file.read()
        encoded_video = base64.b64encode(video_data).decode('utf-8')

    # Emit the complete video to the client
    socketio.emit('video_processed', {'video': encoded_video})
    socketio.emit('video_processed', {'analytics': shots_response_data})


# Home route
@app.route('/')
def home():
    return "Welcome to the Timeout Home Page! What do you want to do?"


# Route to upload a video file
@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 201
        else:
            return jsonify({"error": "File type not allowed"}), 400
    except FileNotFoundError as e:
        print(f'Internal server error while uploading file :- {e}')


# Route to process a video file (e.g., extract frames)
@app.route('/process/<filename>', methods=['GET'])
def process_video(filename):
    file_path = os.path.join(upload_folder, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    param_list = ['attempts', 'dribble_count']  # request.json
    # Process frame asynchronously
    thread = Thread(target=analyze_parameters, args=(file_path, param_list))
    thread.start()
    return jsonify({"message": "File received and processing started"}), 200


# Route to receive video frames
@app.route('/upload_stream', methods=['POST'])
def upload_stream():
    if 'frame' not in request.files:
        return jsonify({"error": "No frame part"}), 400

    file = request.files['frame']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read frame as a byte array
    frame_bytes = file.read()

    # Convert byte array to numpy array
    np_arr = np.frombuffer(frame_bytes, np.uint8)

    # Decode numpy array to image
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Failed to decode frame"}), 400

    # Save frame to disk (optional)
    frame_count = len(os.listdir(frame_upload_folder))
    frame_filename = os.path.join(frame_upload_folder, f'frame_{frame_count}.jpg')
    cv2.imwrite(frame_filename, frame)

    # TODO pass the frame to analysis function
    output_frame = frame  # some_func(frame)
    output_frame = b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n'
    return Response(output_frame, mimetype='multipart/x-mixed-replace; boundary=frame')


# Route to serve a video file which was already analysed
@app.route('/view/<filename>', methods=['GET'])
def serve_video(filename):
    # TODO here we can take video_name/id of previously analyzed from UI
    video_path = os.path.join(output_folder, filename)
    return send_file(video_path, mimetype='video/mp4')


# Route to stream video from the camera
@app.route('/stream')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Generator function to stream video frames
def generate_frames():
    camera = cv2.VideoCapture(0)  # Change to your video source, e.g., 'path/to/video.mp4'

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
