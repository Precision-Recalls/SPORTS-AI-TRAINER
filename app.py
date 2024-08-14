from enum import Enum
from threading import Thread
import logging

from flask import Flask, request, jsonify

from common.azure_storage import upload_blob
from common.utils import load_config
from resources.basketball_res import analyze_basketball_parameters
from resources.yoga_res import analyze_yoga_video

app = Flask(__name__)
logger = logging.Logger('CRITICAL')

config = load_config('configs/config.ini')
allowed_extensions = eval(config['constants']['allowed_extensions'])


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


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
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
    except Exception as e:
        logger.error(f"There is some error with video uploading :- {e}")


@app.route('/process', methods=['POST'])
def process_video():
    try:
        data = request.json
        filename = data['filename']
        param_list = data['param_list']
        drill_type = data['drill_type']

        if drill_type == DrillType.Yoga.value:
            thread = Thread(target=analyze_yoga_video, args=(filename, param_list))
            thread.start()
        elif drill_type == DrillType.BasketBall.value:
            thread = Thread(target=analyze_basketball_parameters, args=(filename, param_list))
            thread.start()

        return jsonify({"message": "File received and processing started"}), 200
    except Exception as e:
        logger.error(f"There is some error with video processing :- {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
