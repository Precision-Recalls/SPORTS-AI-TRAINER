import cv2
import math
import numpy as np
from basketball_analytics.player_class import Player
from common.utils import display_angles, scale_text
from basketball_analytics.shot_detection_utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, \
    clean_ball_pos


class ShotDetector:
    def __init__(self, model, pose_model, class_names, video_link, body_index):
        self.model = model
        self.pose_model = pose_model
        self.class_names = class_names
        self.cap = cv2.VideoCapture(video_link)
        self.body_index = body_index

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None
        self.prev_left_ankle_y = None
        self.prev_right_ankle_y = None
        self.step_threshold = 12
        self.min_wait_frames = 8
        self.wait_frames = 0
        self.makes = 0
        self.attempts = 0

        self.step_counter = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break

            object_detection_results = self.model(self.frame, conf=0.7, iou=0.4, stream=True)
            pose_results = self.pose_model(self.frame, verbose=False, conf=0.7, stream=True)
            step_counter = 0
            if pose_results:
                player = Player(self.frame, pose_results, self.body_index)
                steps = player.count_steps()
                step_counter += steps  # type: ignore
                elbow_angles = player.calculate_elbow_angles()
                display_angles(self.frame, elbow_angles)

                # Annotate the frame with the step count
                text, position, font_scale, thickness = scale_text(self.frame, f"Steps: {step_counter}", (10, 30), 1, 2)
                cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

            for r in object_detection_results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Only create ball points if high confidence or near hoop
                    if (conf > .3 or (
                            in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "ball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))

                    # Create hoop points if high confidence
                    if conf > .5 and current_class == "basket":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1

            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)  # type: ignore

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)  # type: ignore

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    is_goal, description = score(self.ball_pos, self.hoop_pos)

                    # If it is a make, put a green overlay
                    if is_goal:
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames

                    # If it is a miss, put a red overlay
                    else:
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames

    def display_score(self):
        cv2.putText(self.frame, f"Shots:{self.attempts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                    3)  # type: ignore
        cv2.putText(self.frame, f"Goals:{self.makes}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                    3)  # type: ignore

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha,
                                         0)  # type: ignore
            self.fade_counter -= 1
