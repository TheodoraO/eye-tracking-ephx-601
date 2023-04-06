import cv2

# Load video file
cap = cv2.VideoCapture('EyeTracking.mp4')

# Set crop parameters for the video
x, y, w, h = 600, 58, 850, 270

roi = frame[y:y + h, x:x + w]

# Convert the frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the cropped frame
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # Detect circles in the cropped frame using the HoughCircles method
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10,
                               maxRadius=30)
    for circle in circles[0]:
        # Extract the center coordinates and radius of the circle
        center = (circle[0] + x, circle[1] + y)
        radius = circle[2]
        cv2.circle(frame,center,radius, (0,255, 0), 2)
        left_eye =  center
        right_eye = center
    cv2.imshow('frame', frame)
    if cv2.waitkey(1) == ord('q')
        break

cap.release()
cv2.destroyAllWindows