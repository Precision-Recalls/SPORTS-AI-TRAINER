FROM python:3.11

# Install the required dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set up your app
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

# Run the app
CMD ["gunicorn", "--bind=0.0.0.0:8000", "app:app"]
