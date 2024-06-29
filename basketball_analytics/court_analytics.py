import cv2
import numpy as np
from matplotlib import pyplot as plt


def detect_court(file_path):
    # Load the image
    image = cv2.imread(file_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    print(lines)

    # Draw the lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Load the template image of the hoop
    hoop_template = cv2.imread(r'C:\Users\Abhay\Desktop\Precision_Recalls\TimeOut\basketball_hoop_cutout'
                               r'-r66fdbc68cd424847853f89913dad51a2_x7saw_8byvr_540.jpg', 0)
    w, h = hoop_template.shape[::-1]

    # Perform template matching
    res = cv2.matchTemplate(gray, hoop_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    # Draw rectangle around detected hoops
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    # Display the image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()

    # # Show the final result
    # cv2.imshow('Basketball Court Analysis', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    file_path = '../assets/sample_images/sample_image.jpg'
    detect_court(file_path)
