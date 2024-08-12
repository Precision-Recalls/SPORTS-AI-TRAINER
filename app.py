import os
from threading import Thread
from azure.storage.blob import BlobServiceClient
import io
import cv2
import numpy as np
from flask import Flask, send_file, Response, request, jsonify
from flask_socketio import SocketIO
from enum import Enum
from common.utils import load_config
from resources.basketball_res import analyze_basketball_parameters
from resources.yoga_res import analyze_yoga_video
from common.azure_storage import upload_blob
import tempfile
app = Flask(__name__)

config = load_config('configs/config.ini')
allowed_extensions = eval(config['constants']['allowed_extensions'])

upload_folder = config['paths']['upload_folder']
# Directory to save frames
frame_upload_folder = config['paths']['frame_upload_folder']
# Ensure the upload folder exists
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(frame_upload_folder, exist_ok=True)
# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")


class DrillType(Enum):
    Yoga = 'yoga'
    BasketBall = 'basketball'
    Others = 'others'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# Home route
@app.route('/')
def home():
    return "Welcome to the Timeout Home Page! What do you want to do?"


# Route to upload a video file
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        file_data = file.read()
        upload_blob(file.filename, file_data)
        return jsonify({"message": "File uploaded successfully", "file_name": file.filename}), 201
    else:
        return jsonify({"error": "File type not allowed"}), 400


# Route to process a video file (e.g., extract frames)
@app.route('/process', methods=['POST'])
def process_video():
    data = request.json
    filename = data['filename']
    param_list = data['param_list']  # ['attempts', 'dribble_count']
    drill_type = data['drill_type']  # 'yoga','basketball'
    # file_path = os.path.join(upload_folder, filename)

    # if not os.path.exists(file_path):
    #     return jsonify({"error": "File not found"}), 404

    if drill_type == DrillType.Yoga.value:
        thread = Thread(target=analyze_yoga_video, args=(filename, param_list))
        thread.start()
    elif drill_type == DrillType.BasketBall.value:
        thread = Thread(target=analyze_basketball_parameters, args=(filename, socketio, param_list))
        thread.start()
    else:
        # TODO we can add more drill types here
        pass
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
    blob_service_client = BlobServiceClient.from_connection_string(config['azure']['connection_string'])
    container_client = blob_service_client.get_container_client(config['azure']['container_name'])
    blob_client = container_client.get_blob_client(f"processed_{filename}")

    video_data = blob_client.download_blob().readall()
    video_stream = io.BytesIO(video_data)
    
    return send_file(
        video_stream,
        mimetype='video/mp4',
        as_attachment=True,
        download_name=f"processed_{filename}")


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
