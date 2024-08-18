import logging
from enum import Enum
from threading import Thread

from flask import Flask, request, jsonify

from common.azure_storage import upload_blob
from common.utils import load_config
from resources.basketball_res import analyze_basketball_parameters
from resources.fitness_res import analyze_fitness_video
from resources.yoga_res import analyze_yoga_video

app = Flask(__name__)
logger = logging.Logger('CRITICAL')

config = load_config('configs/config.ini')
allowed_extensions = eval(config['constants']['allowed_extensions'])


class DrillType(Enum):
    Yoga = 'yoga'
    BasketBall = 'basketball'
    Fitness = 'fitness'
    Others = 'others'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# Home route
@app.route('/')
def home():
    return "Welcome to the Timeout Home Page! What do you want to do?"


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
    drill_type = data['drill_type']  # 'yoga','basketball', 'fitness'
    drill_name = data['drill_name']  # 'pull_up_bar', 'dead lift'
    
    if drill_type == DrillType.Yoga.value:
        thread = Thread(target=analyze_yoga_video, args=(filename,))
        thread.start()
    elif drill_type == DrillType.BasketBall.value:
        thread = Thread(target=analyze_basketball_parameters, args=(filename,))
        thread.start()
    elif drill_type == DrillType.Fitness.value:
        thread = Thread(target=analyze_fitness_video, args=(filename, drill_name))
        thread.start()
    return jsonify({"message": "File received and processing started"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
