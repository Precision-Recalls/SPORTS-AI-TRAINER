import math
import numpy as np

# Detects if the ball is below the net - used to detect shot attempts
def detect_down(ball_pos, hoop_pos):
    y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    if ball_pos[-1][0][1] > y:
        return True
    return False


# Detects if the ball is around the backboard - used to detect shot attempts
def detect_up(ball_pos, hoop_pos):
    x1 = hoop_pos[-1][0][0] - 4 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 4 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 2 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1]

    if x1 < ball_pos[-1][0][0] < x2 and y1 < ball_pos[-1][0][1] < y2 - 0.5 * hoop_pos[-1][3]:
        return True
    return False


# Checks if center point is near the hoop
def in_hoop_region(center, hoop_pos):
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]

    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


# Removes inaccurate data points
def clean_ball_pos(ball_pos, frame_count):
    # Removes inaccurate ball size to prevent jumping to wrong ball
    if len(ball_pos) > 1:
        # Width and Height
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        # X and Y coordinates
        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        # Frame count
        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)

        # Ball should not move a 4x its diameter within 5 frames
        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()

        # Ball should be relatively square
        elif (w2*1.4 < h2) or (h2*1.4 < w2):
            ball_pos.pop()

    # Remove points older than 30 frames
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 30:
            ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    # Prevents jumping from one hoop to another
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]

        f_dif = f2-f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        # Hoop should not move 0.5x its diameter within 5 frames
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        # Hoop should be relatively square
        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop_pos.pop()

    # Remove old points
    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos

def score1(ball_pos, hoop_pos):
    x = []
    y = []
    rim_height = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]
 
    radius = (sum(ball[3] for ball in ball_pos) / len(ball_pos))/2
    count=0
    
    # For loop to count the number of times the ball touches the rim
    for i in reversed(range(len(ball_pos))):
        # 5 here is the margin of error for the ball to touch the rim
        if rim_height-5<ball_pos[i][0][1]+radius<rim_height+5:
            count+=1
    
    backboard_pos_y_ends=[hoop_pos[-1][0][1] - 4 * hoop_pos[-1][3], hoop_pos[-1][0][1] + 2 * hoop_pos[-1][3]]
    # backboard_pos_x_ends=[hoop_pos[-1][0][0] - 4 * hoop_pos[-1][2], hoop_pos[-1][0][0] + 4 * hoop_pos[-1][2]]
    
    backboard_touch=False
    max_ball_x=0
    
    for i in reversed(range(len(ball_pos))): 

        # if the ball lies in the y-axis range of the backbaord(backboard_pos) and above the rim         
        # and if x-axis coordinate of the ball decreases ones after it was increasing then, it is
        # counted as the touch on the backboard 

        if ball_pos[i][0][1] < rim_height and (ball_pos[i][0][1] < backboard_pos_y_ends[1] and ball_pos[i][0][1] > backboard_pos_y_ends[0]):
            if max_ball_x > ball_pos[i][0][0]:
                backboard_touch=True
                break

            max_ball_x=max(max_ball_x,ball_pos[i][0][0])

            
    # Get first point above rim and first point below rim
    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < rim_height:
            x.append(ball_pos[i][0][0])
            y.append(ball_pos[i][0][1])
            x.append(ball_pos[i+1][0][0])
            y.append(ball_pos[i+1][0][1])
            break

    # Create line from two points
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        print(x, y)
        # Checks if projected line fits between the ends of the rim {x = (y-b)/m}
        predicted_x = ((hoop_pos[-1][0][1] - 0.5*hoop_pos[-1][3]) - b)/m
        rim_x1 = hoop_pos[-1][0][0] - 0.4 * hoop_pos[-1][2]
        rim_x2 = hoop_pos[-1][0][0] + 0.4 * hoop_pos[-1][2]
        
       
        # Case 1: Clean goal, either directly or after touching the rim

        if rim_x1<predicted_x-radius and rim_x2>predicted_x+radius:
            if count>=3:
                return True,"Goal after Touching the rim"
            else: 
                return True,"Clean Goal without touching the rim"

        # Case 2: Goal after touching the rim
        #elif rim_x1<predicted_x<rim_x2:
            return True, "Goal after Touching the rim"

        # Case 3: No goal, either directly or after touching the rim
        elif rim_x2<predicted_x-radius or rim_x1>predicted_x+radius:
            if count>=3:
                return False,"Ball touching the rim and No goal"
            else:
                return False,"No Goal without touching the rim"

        # Case 4: No goal after touching the rim
        #else:
            return False,"Ball touching the rim and No goal"
            
        # elif backboard_touch is True:
        #     # Case 1: Clean goal, either directly or after touching the rim
        #     if rim_x1<predicted_x-radius and rim_x2>predicted_x+radius:
        #         if count>=3:
        #             return True,"Goal after Touching the rim and backboard"
        #         else: 
        #             return True,"Goal after touching the backboard"
            
        #     # Case 2: Goal after touching the rim
        #     elif rim_x1<predicted_x<rim_x2:
        #         return True, "Goal after Touching the rim and backboard"
            
        #     # Case 3: No goal, either directly or after touching the rim
        #     elif rim_x2<predicted_x-radius or rim_x1>predicted_x+radius:
        #         if count>=3:
        #             return False,"No Goal after touching the rim and backboard"
        #         else:
        #             return False,"No Goal after touching backboard"
                
        #     # Case 4: No goal after touching the rim and backboard
        #     else:
        #         return False,"No goal after touching the rim and backboard"
            
        