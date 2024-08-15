This repo contains all the methods for analysing sports videos. We have started with yoga, basketball and some gym exercises.

# Introduction 
This repo is implementation of ML Model for analysing gym drills

# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
    - pip install --upgrade virtualenv 
    - virtualenv -p python3 timeoutenv 
    - source ./timeoutenv/bin/activate
    - pip install -r requirements.txt
    - python manage.py migrate
    - python manage.py runserver
<br />
2.	Software dependencies
    Following installations are needed to run the repository
    - a. Python 3.8.0
<br />
3.	API references
    - filename = path/to/video.mp4
    - url = 'http://localhost:8000/drills/drillname'
    - f = open(filename,'rb'); files = {'video': f};
    - import requests; r = requests.post(url, files=files);
    - r.json() # will give you the response
<br />
<br />
    Drillname can be from the following list:
    - pull_up_bar
    - dumbbell_skull_crusher
    - dumbbell_hammer_curl
    - dumbbell_incline_chest_press
    - dumbbell_stepping_lunge
    - deadlift
    - mountain_climber
    - cable_oblique_twist
    - ab_rollout
    - recovery_shoulder_dislocation
    - bent_over_barbell
    - barbell_bench_press
    - barbell_close_grip_bench_press

The response from the API is a json object and contains 3 keys:
- video: contains the video file in base64 byte converted into string. You can read the file using "import base64" lib
- drill_features: contains a dict object with the format 
    {
        'direction': [0 to n] # even numbers are forward, odd are backward
        'rep': [0 to n] # each number represent the rep count at each frame
    }
- derived_features: contains features calculated during the analysis

    {
        'max_angles': {rep wise max angles},
        'min_angles': {rep wise min angles},
        'rep_time': {rep wise time taken in each rep},
        'rep_completion': {rep wise completion percentage},
        'total_time': integer # total time taken in the drill,
        'total_reps': integer # total reps completed in the drill
    }