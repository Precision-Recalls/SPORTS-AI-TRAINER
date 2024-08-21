from fitness_analytics.constants.body import Body
import mediapipe as mp

mp_pose = mp.solutions.pose


class Connections:
    body_indexes = Body.get_body_indexes()
    connects = []
    for a, b in mp_pose.POSE_CONNECTIONS:
        if a in body_indexes and b in body_indexes:
            connects.append((Body.old_to_new_mapping[a], Body.old_to_new_mapping[b]))
        else:
            continue

    values = frozenset(connects)
