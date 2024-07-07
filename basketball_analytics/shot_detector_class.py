import math
import cv2
import numpy as np
from enum import Enum

from common.utils import scale_text


class shotResult(Enum):
    Goal_Rim_Touch = 'Goal Touching the rim'
    Goal_Rim_No_Touch = 'Goal without touching the rim'
    No_Goal_Rim_Touch = 'Ball touching the rim but No goal'
    No_Goal_No_Rim_Touch = 'No Goal no touching the rim'
    Insufficient_Information = 'Not sufficient data points'


class ShotDetector:
    def __init__(self, class_names, body_index, frame_rate):
        self.class_names = class_names
        self.body_index = body_index
        self.frame_rate = frame_rate
        self.release_detected = False
        self.pose_results = None
        self.detection_results = None
        self.last_shot_description = ""
        self.frame_count = 0
        self.frame = None
        # Shots related variables
        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.player_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.center = None
        self.makes = 0
        self.attempts = 0
        self.last_attempt_count = -1
        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        # Initialize variables to store the previous position of the basketball
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None
        self.dribble_count = 0
        # self.release_angle=None
        # self.steps_before_shot_array = []
        self.current_release_angle = None
        self.current_level_of_player = None
        self.current_player_distance_from_basket = None
        self.current_shot_speed = None
        self.current_shot_power = None

        self.release_frame = None
        self.shot_times = []  # List to store time taken for each shot
        self.frame_steps = [0]
        self.frame_dribble = [0]
        self.step_counter = 0

        # Threshold for the y-coordinate change to be considered as a dribble
        self.dribble_threshold = 3
        self.individual_shot_data = {}

    def run(self, frame_count, frame, step_counter, object_detection_results, pose_results):
        self.frame_count = frame_count
        self.frame = frame
        self.pose_results = pose_results
        self.detection_results = object_detection_results
        self.step_counter = step_counter
        self.individual_shot_data = {}

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
                self.center = (int(x1 + w / 2), int(y1 + h / 2))

                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Only create ball points if high confidence or near hoop
                if current_class == 'ball':
                    if (conf > .3 or (
                            self.in_hoop_region() and conf > 0.15)):
                        self.ball_pos.append((self.center, self.frame_count, w, h, conf))
                        self.calculate_release_parameters()
                        # Update the dribble count
                        self.update_dribble_count()

                # Create hoop points if high confidence
                if conf > .5 and current_class == "basket":
                    self.hoop_pos.append((self.center, self.frame_count, w, h, conf))
                if conf > 0.5 and current_class == "person":
                    self.player_pos.append((self.center, self.frame_count, w, h, conf))
        self.clean_motion()
        self.shot_detection()
        self.display_score()
        return self.frame, self.individual_shot_data

    def calculate_release_parameters(self):
        try:
            rounded_pose_results = np.round(self.pose_results[0].keypoints.data.numpy(), 1)
            # Get the key points for the body parts
            right_elbow = rounded_pose_results[0][self.body_index["right_elbow"]][:2]
            right_wrist = rounded_pose_results[0][self.body_index["right_wrist"]][:2]

            # Calculate the radius of the ball
            radius = (sum(ball[3] for ball in self.ball_pos) / len(self.ball_pos)) / 2

            x_centre, y_centre = self.ball_pos[-1][0]
            right_wrist_array = np.array(rounded_pose_results[0][self.body_index["right_wrist"]][:2])

            # left_wrist = np.array(rounded_pose_results[0][self.body_index["left_wrist"]][:2])
            ball_position = np.array([x_centre, y_centre])

            ball_above_elbow = y_centre < right_elbow[1]
            ball_within_distance = (6 * radius) > np.linalg.norm(ball_position - right_wrist_array) >= (3 * radius)

            # Check if the ball is above the left elbow and within a certain distance from the right wrist
            if not self.release_detected and ball_above_elbow and ball_within_distance:
                # Calculate the release angle while throwing
                ball_pos_x, ball_pos_y = self.ball_pos[-1][0]
                dx = ball_pos_x - right_wrist[0]
                dy = ball_pos_y - right_wrist[1]
                release_angle = np.degrees(np.arctan2(dy, dx))

                # calculate player level from hoop while throwing
                player_top_y_coordinate = self.player_pos[-1][0][1] - (self.player_pos[-1][3] / 2)
                hoop_top_y_coordinate = self.hoop_pos[-1][0][1] - (self.hoop_pos[-1][3] / 2)
                player_level = (player_top_y_coordinate - hoop_top_y_coordinate)

                # calculate player's distance from basket while throwing
                player_centre_x = self.player_pos[-1][0][0]
                player_width = self.player_pos[-1][2]
                hoop_center_x = self.hoop_pos[-1][0][0]
                hoop_width = self.hoop_pos[-1][2]
                player_distance_from_basket = (
                        (hoop_center_x - player_centre_x) - (player_width / 2) - (hoop_width / 2))
                if release_angle < 0:
                    release_angle = -release_angle
                self.release_detected = True
                self.current_release_angle = release_angle
                self.current_level_of_player = player_level
                self.current_player_distance_from_basket = player_distance_from_basket
                self.release_frame = self.frame_count  # Store the release frame

            if self.release_frame is not None:
                # Get the ball positions after the release frame
                ball_positions_after_release = [pos for pos in self.ball_pos if
                                                pos[1] > self.release_frame and pos[1] - self.release_frame < 3]

                if len(ball_positions_after_release) >= 2:
                    # Calculate the time elapsed between the first and second positions after release
                    time_elapsed = (ball_positions_after_release[1][1] - ball_positions_after_release[0][
                        1]) / self.frame_rate

                    # Calculate the distance traveled by the ball
                    x1, y1 = ball_positions_after_release[1][0]
                    x2, y2 = ball_positions_after_release[0][0]
                    distance_traveled = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                    # Calculate the speed of the ball
                    self.current_shot_speed = distance_traveled / time_elapsed

                    # Calculate the energy of the ball (assuming a constant mass)
                    ball_mass = 0.625  # Mass of a standard basketball in kg
                    self.current_shot_power = (0.5 * ball_mass * self.current_shot_speed ** 2) / time_elapsed

        except Exception as e:
            print(f"Error occurred while calculating release angle: {e}")

    def update_dribble_count(self):
        # Check if this is not the first frame
        x_center, y_center = self.ball_pos[-1][0]

        # Check if this is not the first frame
        if self.prev_y_center is not None:
            # Calculate the change in the y-coordinate of the basketball's center
            delta_y = y_center - self.prev_y_center

            # Check if the basketball's y-coordinate has changed by more than the threshold
            if (
                    self.prev_delta_y is not None
                    and self.prev_delta_y > self.dribble_threshold
                    and delta_y < -self.dribble_threshold
            ):
                rounded_pose_results = np.round(self.pose_results[0].keypoints.data.numpy(), 1)

                # Get the key points for the body parts
                right_wrist = np.array(rounded_pose_results[0][self.body_index["right_wrist"]][:2])
                left_wrist = np.array(rounded_pose_results[0][self.body_index["left_wrist"]][:2])
                ball_position = np.array([x_center, y_center])

                if (
                        np.linalg.norm(ball_position - right_wrist) < 50
                        or np.linalg.norm(ball_position - left_wrist) < 50
                ) and y_center > max(right_wrist[1], left_wrist[1]):
                    # If so, increment the dribble count
                    self.dribble_count += 1

            # Store the change in the y-coordinate for the next frame
            self.prev_delta_y = delta_y

        # Store the current position of the basketball for the next frame
        self.prev_x_center = x_center
        self.prev_y_center = y_center

    def display_score(self):
        # Add text
        text = f"Goals: {str(self.makes)}/{str(self.attempts)}"
        text, position, font_scale, thickness = scale_text(self.frame, text, (10, 30), 2, 4)
        cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Display description of the last shot made
        text, position, font_scale, thickness = scale_text(self.frame, f"Last shot: {self.last_shot_description}",
                                                           (10, 55), 1, 2)
        cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        text, position, font_scale, thickness = scale_text(self.frame, f"Total Dribble Count: {self.dribble_count}",
                                                           (10, 95), 1, 2)
        cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # text, position, font_scale, thickness = scale_text(self.frame, f"Total steps per attempt: {
        # self.frame_steps[-1]-self.frame_steps[-2]}", (10, 95), 1, 2) cv2.putText(self.frame, text, position,
        # cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        if self.current_release_angle is not None:
            text, position, font_scale, thickness = scale_text(self.frame,
                                                               f"Release angle: {self.current_release_angle:.2f}",
                                                               (10, 155), 1, 2)
            cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        if self.current_level_of_player is not None:
            text, position, font_scale, thickness = scale_text(self.frame,
                                                               f"Player's Level From Rim: {self.current_level_of_player:.2f}",
                                                               (10, 175), 1, 2)
            cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        if self.current_player_distance_from_basket is not None:
            text, position, font_scale, thickness = scale_text(self.frame,
                                                               f"Player's Distance From Rim: {self.current_player_distance_from_basket:.2f}",
                                                               (10, 195), 1, 2)
            cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        if self.current_shot_speed is not None:
            text, position, font_scale, thickness = scale_text(self.frame,
                                                               f"Ball Speed : {self.current_shot_speed:.2f}",
                                                               (10, 115), 1, 2)
            cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        if self.current_shot_power is not None:
            text, position, font_scale, thickness = scale_text(self.frame,
                                                               f"Ball Power : {self.current_shot_power:.2f}",
                                                               (10, 135), 1, 2)
            cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Display shot times for each shot
        if self.shot_times:
            frame_width = self.frame.shape[1]
            frame_height = self.frame.shape[0]
            x_position = frame_width - 270  # Adjust the x-position as needed
            y_position = 30  # Adjust the y-position as needed

            for i, shot_time in enumerate(self.shot_times):
                text = f"Shot {i + 1} time: {shot_time:.2f} seconds"
                text, _, font_scale, thickness = scale_text(self.frame, text, (x_position, y_position), 1, 2)
                cv2.putText(self.frame, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                            thickness)
                y_position += 20  # Increase the y-position for the next shot time

        # # Display shot times for each shot
        # if self.frame_dribble:
        #     frame_width = self.frame.shape[1]
        #     frame_height = self.frame.shape[0]
        #     x_position = frame_width - 270  # Adjust the x-position as needed
        #     y_position = 30  # Adjust the y-position as needed
        #     prev_shot_dribbles = self.frame_dribble[0]
        #     for i, dribbles in enumerate(self.frame_dribble[1:]):
        #         text = f"Ball dribbled in {i + 1}th attempt: {dribbles - prev_shot_dribbles}"
        #         text, _, font_scale, thickness = scale_text(self.frame, text, (x_position, y_position), 1, 2)
        #         cv2.putText(self.frame, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        #                     (0, 0, 0),
        #                     thickness)
        #         y_position += 20  # Increase the y-position for the next shot time
        #         prev_shot_dribbles = dribbles

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha,
                                         0)  # type: ignore
            self.fade_counter -= 1

    def clean_motion(self):
        # Clean and display ball motion
        self.clean_ball_pos()
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.clean_hoop_pos()
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.up:
                self.up = self.detect_up()
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = self.detect_down()
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.last_attempt_count = self.attempts
                    self.attempts += 1
                    self.frame_steps.append(self.step_counter)
                    self.frame_dribble.append(self.dribble_count)

                    is_goal, description = self.score()
                    self.last_shot_description = description  # Update last_shot_description

                    if self.release_frame is not None:
                        shot_time = (self.down_frame - self.release_frame) / self.frame_rate
                        self.shot_times.append(shot_time)  # Append the shot time to the list

                    # If it is a make, put a green overlay
                    if is_goal:
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames

                    # If it is a miss, put a red overlay
                    else:
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames
                    self.individual_shot_data = {
                        'attempts': self.attempts,
                        'goals': self.makes,
                        'shot_desc': self.last_shot_description,
                        'dribble_count': self.frame_dribble[-1] - self.frame_dribble[-2],
                        'release_angle': self.current_release_angle,
                        'level_from_rim': self.current_level_of_player,
                        'distance_from_basket': self.current_player_distance_from_basket,
                        'shot_speed': self.current_shot_speed,
                        'shot_power': self.current_shot_power,
                        'shot_time': self.shot_times[-1],
                        'steps': self.frame_steps[-1] - self.frame_steps[-2]
                    }
                    self.up = False
                    self.down = False
                    self.current_release_angle = None
                    self.release_detected = False
                    self.current_level_of_player = None
                    self.current_player_distance_from_basket = None
                    self.current_shot_speed = None
                    self.current_shot_power = None
                    self.release_frame = None  # Reset the release frame

    # Detects if the ball is below the net - used to detect shot attempts
    def detect_down(self):
        return self.ball_pos[-1][0][1] > self.hoop_pos[-1][0][1] + 0.5 * self.hoop_pos[-1][3]

    # Detects if the ball is around the backboard - used to detect shot attempts
    def detect_up(self):
        x1 = self.hoop_pos[-1][0][0] - 4 * self.hoop_pos[-1][2]
        x2 = self.hoop_pos[-1][0][0] + 4 * self.hoop_pos[-1][2]
        y1 = self.hoop_pos[-1][0][1] - 2 * self.hoop_pos[-1][3]
        y2 = self.hoop_pos[-1][0][1]

        return x1 < self.ball_pos[-1][0][0] < x2 and y1 < self.ball_pos[-1][0][1] < y2 - 0.5 * self.hoop_pos[-1][3]

    # Checks if center point is near the hoop
    def in_hoop_region(self):
        if len(self.hoop_pos) < 1:
            return False
        x, y = self.center

        x1 = self.hoop_pos[-1][0][0] - 1 * self.hoop_pos[-1][2]
        x2 = self.hoop_pos[-1][0][0] + 1 * self.hoop_pos[-1][2]
        y1 = self.hoop_pos[-1][0][1] - 1 * self.hoop_pos[-1][3]
        y2 = self.hoop_pos[-1][0][1] + 0.5 * self.hoop_pos[-1][3]

        return x1 < x < x2 and y1 < y < y2

    # Removes inaccurate data points
    def clean_ball_pos(self):
        # Removes inaccurate ball size to prevent jumping to wrong ball
        if len(self.ball_pos) > 1:
            # Width and Height
            w1 = self.ball_pos[-2][2]
            h1 = self.ball_pos[-2][3]
            w2 = self.ball_pos[-1][2]
            h2 = self.ball_pos[-1][3]

            # X and Y coordinates
            x1 = self.ball_pos[-2][0][0]
            y1 = self.ball_pos[-2][0][1]
            x2 = self.ball_pos[-1][0][0]
            y2 = self.ball_pos[-1][0][1]

            # Frame count
            f1 = self.ball_pos[-2][1]
            f2 = self.ball_pos[-1][1]
            f_dif = f2 - f1

            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            max_dist = 4 * math.sqrt(w1 ** 2 + h1 ** 2)

            # Ball should not move a 4x its diameter within 5 frames
            if (dist > max_dist) and (f_dif < 5):
                self.ball_pos.pop()

            # Ball should be relatively square
            elif (w2 * 1.4 < h2) or (h2 * 1.4 < w2):
                self.ball_pos.pop()

        # Remove points older than 30 frames
        if len(self.ball_pos) > 0:
            if self.frame_count - self.ball_pos[0][1] > 30:
                self.ball_pos.pop(0)

    def clean_hoop_pos(self):
        # Prevents jumping from one hoop to another
        if len(self.hoop_pos) > 1:
            x1 = self.hoop_pos[-2][0][0]
            y1 = self.hoop_pos[-2][0][1]
            x2 = self.hoop_pos[-1][0][0]
            y2 = self.hoop_pos[-1][0][1]

            w1 = self.hoop_pos[-2][2]
            h1 = self.hoop_pos[-2][3]
            w2 = self.hoop_pos[-1][2]
            h2 = self.hoop_pos[-1][3]

            f1 = self.hoop_pos[-2][1]
            f2 = self.hoop_pos[-1][1]

            f_dif = f2 - f1

            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

            # Hoop should not move 0.5x its diameter within 5 frames
            if dist > max_dist and f_dif < 5:
                self.hoop_pos.pop()

            # Hoop should be relatively square
            if (w2 * 1.3 < h2) or (h2 * 1.3 < w2):
                self.hoop_pos.pop()

        # Remove old points
        if len(self.hoop_pos) > 25:
            self.hoop_pos.pop(0)

    def score(self):
        x = []
        y = []
        rim_height = self.hoop_pos[-1][0][1] - 0.5 * self.hoop_pos[-1][3]

        radius = (sum(ball[3] for ball in self.ball_pos) / len(self.ball_pos)) / 2
        count = 0

        # For loop to count the number of times the ball touches the rim
        for i in reversed(range(len(self.ball_pos))):
            # 5 here is the margin of error for the ball to touch the rim
            if rim_height - 5 < self.ball_pos[i][0][1] + radius < rim_height + 5:
                count += 1

        # Get first point above rim and first point below rim
        for i in reversed(range(len(self.ball_pos))):
            if self.ball_pos[i][0][1] < rim_height:
                x.append(self.ball_pos[i][0][0])
                y.append(self.ball_pos[i][0][1])
                x.append(self.ball_pos[i + 1][0][0])
                y.append(self.ball_pos[i + 1][0][1])
                break

        # Create line from two points
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            # Checks if projected line fits between the ends of the rim {x = (y-b)/m}
            predicted_x = ((self.hoop_pos[-1][0][1] - 0.5 * self.hoop_pos[-1][3]) - b) / m
            rim_x1 = self.hoop_pos[-1][0][0] - 0.4 * self.hoop_pos[-1][2]
            rim_x2 = self.hoop_pos[-1][0][0] + 0.4 * self.hoop_pos[-1][2]

            # Case 1: Clean goal, either directly or after touching the rim
            if rim_x1 < predicted_x - radius and rim_x2 > predicted_x + radius:
                if count >= 3:
                    return True, shotResult.Goal_Rim_Touch.value
                else:
                    return True, shotResult.Goal_Rim_No_Touch.value

            # Case 2: No goal, either directly or after touching the rim
            elif rim_x2 < predicted_x - radius or rim_x1 > predicted_x + radius:
                if count >= 3:
                    return False, shotResult.No_Goal_Rim_Touch.value
                else:
                    return False, shotResult.No_Goal_No_Rim_Touch.value
        return False, shotResult.Insufficient_Information.value
