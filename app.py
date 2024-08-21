import logging
import os
import sys
from threading import Thread

from flask import Flask, request, jsonify

from common.azure_storage import upload_blob
from common.utils import load_config, create_api_response, DrillType, get_service_bus_connection_obj
from resources.basketball_res import analyze_basketball_parameters
from resources.fitness_res import analyze_fitness_video
from resources.yoga_res import analyze_yoga_video

app = Flask(__name__)
logger = logging.Logger('CRITICAL')

config = load_config('configs/config.ini')
allowed_extensions = eval(config['constants']['allowed_extensions'])
azure_service_bus_connection_string = config['azure']['azure_service_bus_connection_string']
queue_name = config['azure']['queue_name']
sender = get_service_bus_connection_obj(azure_service_bus_connection_string, queue_name)


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
            return create_api_response("No file part", 400)

        file = request.files['file']
        if file.filename == '':
            return create_api_response("No selected file", 400)

        if file and allowed_file(file.filename):
            file_data = file.read()
            upload_blob(file.filename, file_data)
            return create_api_response(f"File :- {file.filename} uploaded successfully", 201)
        else:
            return create_api_response("File type not allowed", 400)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f'There is some issue with file uploading API {exc_tb.tb_lineno}th line '
                     f'in {fname}, error {exc_type}')
        create_api_response("Couldn't upload file due to azure blob service connectivity issue", 400)


# Route to process a video file (e.g., extract frames)
@app.route('/process', methods=['POST'])
def process_video():
    try:
        data = request.json
        filename = data['filename']
        drill_type = data['drill_type']  # 'yoga','basketball', 'fitness'
        drill_name = data['drill_name']  # 'pull_up_bar', 'dead lift'

        if drill_type == DrillType.Yoga.value:
            thread = Thread(target=analyze_yoga_video, args=(filename, sender))
            thread.start()
            logger.info(f"Yoga thread started for {filename} file")
        elif drill_type == DrillType.BasketBall.value:
            thread = Thread(target=analyze_basketball_parameters, args=(filename, sender))
            thread.start()
            logger.info(f"BasketBall thread started for {filename} file")
        elif drill_type == DrillType.Fitness.value:
            thread = Thread(target=analyze_fitness_video, args=(filename, drill_name, sender))
            thread.start()
            logger.info(f"Fitness thread started for {filename} file")
        return jsonify({"message": "File received and processing started"}), 200
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f'There is some issue with file processing API {exc_tb.tb_lineno}th line '
                     f'in {fname}, error {exc_type}')
        return create_api_response("Couldn't process file due to azure webapp issue", 400)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
