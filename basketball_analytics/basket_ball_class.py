import cv2
import math
import numpy as np
from basketball_analytics.player_class import Player
from common.utils import display_angles, scale_text,calculate_angle
from basketball_analytics.shot_detector_class import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, \
    clean_ball_pos


class BasketBallGame:
    def __init__(self, model, pose_model, class_names, video_link, body_index):
        # Essential variables
        self.model = model
        self.pose_model = pose_model
        self.class_names = class_names
        self.cap = cv2.VideoCapture(video_link)
        self.init_video_writer('output_video1.mp4')
        self.body_index = body_index
        self.player = Player(self.body_index)
        self.last_shot_description = ""
        self.frame_count = 0
        self.frame = None
        self.step_counter = 0
        
        
        self.total_steps = 0
        #self.steps_before_shot = 0
        # Shots related variables
        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.player_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.makes = 0
        self.attempts = 0
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
        self.release_angle=0
        # Initialize the dribble counter
        self.dribble_count = 0
        #self.release_angle=None
        #self.steps_before_shot_array = []
        self.current_release_angle = None
        self.current_level_of_player=None

        # Threshold for the y-coordinate change to be considered as a dribble
        self.dribble_threshold = 3

        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                # End of the video or an error occurred
                break

            object_detection_results = self.model(self.frame, conf=0.7, iou=0.4, stream=True)
            pose_results = self.pose_model(self.frame, verbose=False, conf=0.7)
            step_counter = 0
            steps_before_shot = 0
            
            if pose_results:
                # Round the results to the nearest decimal
                rounded_pose_results = np.round(pose_results[0].keypoints.data.numpy(), 1)
                steps = self.player.count_steps(rounded_pose_results)
                steps_before_shot += steps  # Update steps_before_shot
                
                step_counter += steps
                #steps_before_shot += steps
                elbow_angles = self.player.calculate_elbow_angles(rounded_pose_results)
                if elbow_angles:
                    display_angles(self.frame, elbow_angles)

                
                # Annotate the frame with the step count
                text, position, font_scale,thickness = scale_text(self.frame, f"Total Steps Taken: {step_counter}", (10, 75), 1, 2)
                cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                # Annotate the frame with the step count
                # text, position, font_scale,thickness = scale_text(self.frame, f"Steps Taken for the shot: {steps_before_shot}", (10, 105), 1, 2)
                # cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                
                 
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
                    if current_class=='ball':
                        if (conf > .3 or (
                                in_hoop_region(center, self.hoop_pos) and conf > 0.15)):
                            self.ball_pos.append((center, self.frame_count, w, h, conf))
                            self.calculate_release_angle_and_player_level(self.ball_pos,pose_results,self.hoop_pos,self.player_pos)

                        # Update the dribble count
                            self.update_dribble_count(self.ball_pos,pose_results)

                    # Create hoop points if high confidence
                    if conf > .5 and current_class == "basket":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                    if conf>0.5 and current_class=="person":
                        self.player_pos.append((center,self.frame_count,w,h,conf))
    
            
            self.clean_motion()
            self.shot_detection()
            
            self.display_score()
            self.frame_count += 1
            self.video_writer.write(self.frame)

            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        self.cap.release()
        self.video_writer.release()
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

    def calculate_release_angle_and_player_level(self, ball_pos, pose_results,hoop_pos,player_pos):
        try: 
            rounded_pose_results = np.round(pose_results[0].keypoints.data.numpy(), 1)
            # Get the key points for the body parts
            right_elbow = rounded_pose_results[0][self.body_index["right_elbow"]][:2]
            right_wrist = rounded_pose_results[0][self.body_index["right_wrist"]][:2]
            
            # Calculate the radius of the ball
            radius = (sum(ball[3] for ball in ball_pos) / len(ball_pos)) / 2
            x_centre,y_centre = ball_pos[-1][0]
            right_wrist_array = np.array(rounded_pose_results[0][self.body_index["right_wrist"]][:2])
            #left_wrist = np.array(rounded_pose_results[0][self.body_index["left_wrist"]][:2])
            ball_position = np.array([x_centre,y_centre])
            
            ball_above_elbow = y_centre < right_elbow[1]
            ball_within_distance = (6*radius)>np.linalg.norm(ball_position - right_wrist_array) >= (3* radius)
            
             # Initialize release_detected flag
            if not hasattr(self, 'release_detected'):
                self.release_detected = False

            # Check if the ball is above the left elbow and within a certain distance from the right wrist
            if  not self.release_detected and ball_above_elbow and ball_within_distance:
                print("Ball is above the right elbow and within a certain distance from the right wrist")
                # Calculate the release angle
                ball_pos_x, ball_pos_y = ball_pos[-1][0]
                dx = ball_pos_x - right_wrist[0]
                dy = ball_pos_y - right_wrist[1]
                release_angle = np.degrees(np.arctan2(dy, dx))
                
                player_top_y_coordinate = player_pos[-1][0][1]-(player_pos[-1][3]/2)
                hoop_top_y_coordinate = hoop_pos[-1][0][1]-(hoop_pos[-1][3]/2)
                print(f"Player top y coordinate: {player_top_y_coordinate}")
                print(f"Hoop top y coordinate: {hoop_top_y_coordinate}")
                player_level=(player_top_y_coordinate-hoop_top_y_coordinate)
                if release_angle < 0:
                    release_angle=-release_angle
                self.release_detected = True
                self.current_release_angle = release_angle
                self.current_level_of_player=player_level
                    
        except Exception as e:
            print(f"Error occurred while calculating release angle: {e}")
    


    def update_dribble_count(self,ball_pos,pose_results):
        # Check if this is not the first frame
        x_center, y_center = ball_pos[-1][0]
            
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
                # if pose_results and len(pose_results) > 0:
                #     # Round the results to the nearest decimal
                    rounded_pose_results = np.round(pose_results[0].keypoints.data.numpy(), 1)
                    
    
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
                    self.current_release_angle=None
                    self.release_detected=False
                    self.current_level_of_player=None
                    
                    is_goal, description = score(self.ball_pos, self.hoop_pos)
                    self.last_shot_description = description  # Update last_shot_description
                    #print(f"Attempt: {self.attempts} - {description}")
                    print(f"Attempt: {self.attempts} - {description}")
                    
                    # If it is a make, put a green overlay
                    if is_goal:
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames

                    # If it is a miss, put a red overlay
                    else:
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames
                     # Add the current steps to steps_before_shot


    def display_score(self):

        # Add text
        text = f"Goals: {str(self.makes)}/{str(self.attempts)}"
        text, position,font_scale,thickness = scale_text(self.frame,text, (10, 30), 2, 4)
        cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),thickness)
       
        # Display description of the last shot made
        text, position, font_scale, thickness = scale_text(self.frame, f"Last shot: {self.last_shot_description}", (10, 55), 1, 2)
        cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        text, position, font_scale, thickness = scale_text(self.frame, f"Total Dribble Count: {self.dribble_count}", (10, 95), 1, 2)
        cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        if self.current_release_angle is not None:
            text, position, font_scale,thickness = scale_text(self.frame, f"Release angle: {self.current_release_angle:.2f}", (10, 155), 1, 2)
            cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        if self.current_level_of_player is not None:    
            text, position, font_scale,thickness = scale_text(self.frame, f"Player Level From Rim: {self.current_level_of_player:.2f}", (10, 175), 1, 2)
            cv2.putText(self.frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                      
        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha,
                                         0)  # type: ignore
            self.fade_counter -= 1
    
    def init_video_writer(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
