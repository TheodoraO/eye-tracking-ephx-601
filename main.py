import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load video file
cap = cv2.VideoCapture('EyeTracking.mp4')

# Set crop parameters for the video
x, y, w, h = 600, 58, 850, 270

# Define the coordinates of the left and right eyes
left_eye = (671, 178)  # replace with actual coordinates
right_eye = (814, 182)  # replace with actual coordinates


# Define a function to calculate the distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Define a function to detect saccades
def detect_saccades(input_frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

    # Use the new variable names to refer to the eye centers
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # Crop the frame to the region of interest
    roi = gray[y:y + h, x:x + w]

    # Apply Gaussian blur to the cropped frame
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # Detect circles in the cropped frame using the HoughCircles method
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10,
                               maxRadius=30)

    # Draw a 3x3 grid around the left eye
    cv2.rectangle(roi, (left_eye_x - 50, left_eye_y - 50), (left_eye_x + 50, left_eye_y + 50), (255, 0, 0),
                  2)

    # Draw a 3x3 grid around the right eye
    cv2.rectangle(roi, (right_eye_x - 50, right_eye_y - 50), (right_eye_x + 50, right_eye_y + 50),
                  (255, 0, 0), 2)

    # Loop through the detected circles
    for circle in circles[0]:
        # Extract the center coordinates and radius of the circle
        center = (circle[0] + x, circle[1] + y)
        radius = circle[2]

        # Calculate the distance between the center of the circle and the left and right eye centers
        left_eye_distance = distance(center, left_eye)
        right_eye_distance = distance(center, right_eye)

        # Check if the circle is within a certain distance of the left or right eye center
        if left_eye_distance < 50 or right_eye_distance < 50:
            # Draw a green circle around the detected saccade
            cv2.circle(input_frame, center, radius, (0, 255, 0), 2)

            # Pause the video and calculate the number of pixels in both eyes
            cv2.imshow('frame', input_frame)
            cv2.waitKey(0)

            # Calculate the number of pixels in the left eye
            left_eye_pixels = np.sum(roi[left_eye_y - 50:left_eye_y + 50, left_eye_x - 50:left_eye_x + 50])

            # Calculate the number of pixels in the right eye
            right_eye_pixels = np.sum(roi[right_eye_y - 50:right_eye_y + 50, right_eye_x - 50:right_eye_x + 50])

            # Display the number of pixels in both eyes using matplotlib
            fig, ax = plt.subplots()
            ax.bar(['Left Eye', 'Right Eye'], [left_eye_pixels, right_eye_pixels])
            plt.show()

    # Loop through frames in video
    while True:
        # Read next frame
        ret, frame = cap.read()
        if not ret:
            break
        detect_saccades(frame)

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
