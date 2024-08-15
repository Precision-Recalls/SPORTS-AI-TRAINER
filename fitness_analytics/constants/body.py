class Body:
    right_shoulder = 12
    left_shoulder = 11
    right_elbow = 14
    left_elbow = 13
    right_wrist = 16
    left_wrist = 15
    right_hip = 24
    left_hip = 23
    right_knee = 26
    left_knee = 25
    right_ankle = 28
    left_ankle = 27

    def get_body_indexes():
        return [
            Body.right_shoulder,
            Body.left_shoulder,
            Body.right_elbow,
            Body.left_elbow,
            Body.right_wrist,
            Body.left_wrist,
            Body.right_hip,
            Body.left_hip,
            Body.right_knee,
            Body.left_knee,
            Body.right_ankle,
            Body.left_ankle
        ]

    newBody_dict = {
        'right_shoulder': 0,
        'left_shoulder': 1,
        'right_elbow': 3,
        'left_elbow': 4,
        'right_wrist': 5,
        'left_wrist': 6,
        'right_hip': 7,
        'left_hip': 8,
        'right_knee': 9,
        'left_knee': 10,
        'right_ankle': 11,
        'left_ankle': 12
    }

    old_to_new_mapping = {
        12: 0,
        11: 1,
        14: 2,
        13: 3,
        16: 4,
        15: 5,
        24: 6,
        23: 7,
        26: 8,
        25: 9,
        28: 10,
        27: 11
    }
