import os

# Correctness Configs
# robustness
BUFFER_RATIO = 0.10
FRAME_SUBSET = 8  # only even numbers

# smoothing
WINDOW_LENGTH = 13
POLYORDER = 2
ROLLING_WINDOW = 9
RANGE_RATIO = 0.6
# --------------------------------------------

# Draw Configs
BODY_COLOR_BGR = (212, 231, 235)
STYLE = 'dashed'
GAP = 8
CONNECTION_LINE_THICKNESS = 2

# filtering lines
PRESENCE_THRESHOLD = 0.5
VISIBILITY_THRESHOLD = 0.5
# -----------------------------------------------

# Drill Configs
EXCERCISES = dict(
    pull_up_bar=dict(
        body_part='elbow',
        complements=['shoulder', 'wrist'],
        range=[90, 160]
    ),
    dumbbell_skull_crusher=dict(
        body_part='elbow',
        complements=['shoulder', 'wrist'],
        range=[105, 170]
    ),
    dumbbell_hammer_curl=dict(
        body_part='elbow',
        complements=['shoulder', 'wrist'],
        range=[50, 165]
    ),
    dumbbell_incline_chest_press=dict(
        body_part='elbow',
        complements=['shoulder', 'wrist'],
        range=[100, 165]
    ),
    dumbbell_stepping_lunge=dict(
        body_part='knee',
        complements=['hip', 'ankle'],
        range=[110, 160]
    ),
    deadlift=dict(
        body_part='hip',
        complements=['shoulder', 'knee'],
        range=[100, 150]
    ),

    mountain_climber=dict(
        body_part='knee',
        complements=['hip', 'ankle'],
        range=[50, 165]
    ),

    cable_oblique_twist=dict(
        body_part='elbow',
        complements=['shoulder', 'wrist'],
        range=[50, 165]
    ),
    bent_over_barbell=dict(
        body_part='elbow',
        complements=['shoulder', 'wrist'],
        range=[90, 170]
    ),
    barbell_bench_press=dict(
        body_part='elbow',
        complements=['shoulder', 'wrist'],
        range=[100, 150]
    ),
    barbell_close_grip_bench_press=dict(
        body_part='elbow',
        complements=['shoulder', 'wrist'],
        range=[80, 160]
    ),
    ab_rollout=dict(
        body_part='knee',
        complements=['hip', 'ankle'],
        range=[40, 120]
    ),
    recovery_shoulder_dislocation=dict(
        body_part='left_elbow',
        complements=['left_shoulder', 'left_wrist'],
        range=[100, 160]
    )
)

# ----------------------------------------------------

# Model Configs
STATIC_IMAGE_MODE = False
MODEL_COMPLEXITY = 1
SMOOTH_LANDMARK = True
MIN_DETECTION_CONFIDENCE = 0.60
MIN_TRACKING_CONFIDENCE = 0.60

# -----------------------------------------------------

# Video Configs

# Dimensions for the processed video
WIDTH = 1260  # width of video created
HEIGHT = 720  # height of video created
SKIP_FRAME = 1  # Number of frames skipped while processing video
DOWNSIZE_FACTOR = 3  # Reduced size factor of frame before sent to ML model for inference

# Location of video and video writer
parent_file_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(parent_file_dir, 'data')
VIDEO_PATH = os.path.join(DATA_PATH, 'drills')
VIDEO_WRITER_PATH = os.path.join(DATA_PATH, 'solutions')
