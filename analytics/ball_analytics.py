import torch


def detect_ball(frames):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    ball_positions = []

    for frame in frames:
        results = model(frame)
        for *xyxy, conf, cls in results.xyxy[0]:
            if cls == 0:  # Assuming class 0 is the basketball
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                ball_positions.append((x_center, y_center))

    return ball_positions


def classify_throw(ball_positions, hoop_position, backboard_y):
    if not ball_positions:
        return "No throw detected"

    final_position = ball_positions[-1]
    x, y = final_position

    hoop_radius = 0.1  # Adjust based on your coordinate system

    if (x - hoop_position[0]) ** 2 + (y - hoop_position[1]) ** 2 < hoop_radius ** 2:
        return "Clean inside hoop"
    elif abs(x - hoop_position[0]) < hoop_radius and abs(y - hoop_position[1]) < hoop_radius:
        return "Inside hoop touching rim"
    elif y < backboard_y:
        return "Inside hoop reflecting from backboard"
    elif (x - hoop_position[0]) ** 2 + (y - hoop_position[1]) ** 2 < 1.5 * hoop_radius ** 2:
        return "On the rim"
    else:
        return "Outside the hoop"


def calculate_throw_time(ball_positions, frame_rate):
    if not ball_positions:
        return 0

    num_frames = len(ball_positions)
    throw_time = num_frames / frame_rate

    return throw_time
