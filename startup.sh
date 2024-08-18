#!/bin/bash

apt-get update && apt-get install -y libgl1-mesa-glx

EXPOSE 5000

# Start the Gunicorn server
exec gunicorn --bind 0.0.0.0:5000 app:app
