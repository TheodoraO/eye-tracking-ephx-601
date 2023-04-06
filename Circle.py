import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the eye image
img = Image.open("eye_image.png")

# Convert the image to a numpy array
img_array = np.array(img)

# Convert the image to grayscale
gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect circles in the image
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# Draw the detected circles on the original image
output = img_array.copy()
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)

# Display the image with detected circles
plt.imshow(output)
plt.show()
