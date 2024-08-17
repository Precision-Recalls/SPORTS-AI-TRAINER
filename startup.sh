# startup.sh
#!/bin/bash

# Install required libraries
apt-get update && apt-get install -y libgl1-mesa-glx

# Start your Flask or any other service
gunicorn --bind=0.0.0.0 --timeout 600 app:app
