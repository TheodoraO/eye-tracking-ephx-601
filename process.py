import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard

# Load video file
cap = cv2.VideoCapture('EyeTracking.mp4')

# Set crop parameters for the video
x, y, w, h = 550, 58, 750, 260

# Set up grid to track monkey gaze
grid_size = 50
num_cols = int(w / grid_size)
num_rows = int(h / grid_size)
grid = np.zeros((num_rows, num_cols), dtype=int)

# Set up circle detection parameters
min_radius = 20
max_radius = 60
circle_threshold = 0.5

# Set up figure to display video and grid
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim([0, w])
ax.set_ylim([0, h])
plt.xticks(np.arange(0, w, grid_size))
plt.yticks(np.arange(0, h, grid_size))
plt.grid(linewidth=0.5)

# Calculate center of the eye region
eye_center_x = x + w // 2
eye_center_y = y + h // 2

# Define ROIs for straight ahead and upper right
straight_ahead_roi = np.zeros((2, 2))
straight_ahead_roi[0, 0] = 0
straight_ahead_roi[0, 1] = num_cols // 2
straight_ahead_roi[1, 0] = 2
straight_ahead_roi[1, 1] = num_cols // 2 + 2
upper_right_roi = np.zeros((2, 2))
upper_right_roi[0, 0] = 0
upper_right_roi[0, 1] = num_cols
upper_right_roi[1, 0] = 0
upper_right_roi[1, 1] = 2

# Initialize previous grid for comparison
prev_grid = np.zeros((num_rows, num_cols), dtype=int)

# Initialize variables for pausing/unpausing
paused = False

# Loop through frames in video
while cap.isOpened():
    while paused:
        if keyboard.is_pressed('p'):
            paused = False
    # Read next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Crop frame to focus on monkey's face
    frame = frame[y:y + h, x:x + w]

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, circle_threshold, 20,
                               param1=50, param2=30,
                               minRadius=min_radius, maxRadius=max_radius)

    # Process circles if any are detected
    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Loop through detected circles
        for i in circles[0, :]:
            # Get circle center coordinates
            x = i[0]
            y = i[1]

            # Determine which grid cell the circle falls in
            col = int((x - eye_center_x) / grid_size) + num_cols // 2
            row = int((y - eye_center_y) / grid_size) + num_rows // 2
            if 0 <= row < num_rows and 0 <= col < num_cols:
                grid[row, col] += 1
                cv2.circle(frame, (x, y), i[2], (0, 0, 255), 2)
                rect = plt.Rectangle((col * grid_size, row * grid_size),
                                     grid_size, grid_size, linewidth=0.5,
                                     edgecolor='k', facecolor='none')
                ax.add_patch(rect)

                # Calculate pupil position with respect to open eye circles
            pupil_position = np.array([circles[0, 0, 0], circles[0, 0, 1]])
            for i in range(1, len(circles[0])):
                circle_center = np.array([circles[0, i, 0], circles[0, i, 1]])
                if np.linalg.norm(circle_center - pupil_position) > circles[0, i, 2]:
                    pupil_position = circle_center

            # Determine which ROI the pupil is in
            if (straight_ahead_roi[0, 0] <= pupil_position[1] / grid_size <= straight_ahead_roi[0, 1] and
                    straight_ahead_roi[1, 0] <= pupil_position[0] / grid_size <= straight_ahead_roi[1, 1]):
                pupil_roi = 'straight ahead'
            elif (upper_right_roi[0, 0] <= pupil_position[1] / grid_size <= upper_right_roi[0, 1] and
                  upper_right_roi[1, 0] <= pupil_position[0] / grid_size <= upper_right_roi[1, 1]):
                pupil_roi = 'upper right'
            else:
                pupil_roi = 'directly at the screen'

            # Determine if saccades has occurred
            if not np.array_equal(grid, prev_grid):
                saccade = True
            else:
                saccade = False

            # Print information for current frame
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f'On frame {frame_num}, the monkey is looking {pupil_roi} and '
                  f'{"a saccade has occurred" if saccade else "no saccade has occurred"}')

            # Update previous grid
            prev_grid = np.copy(grid)

        # Show frame with circles drawn on it
        cv2.imshow('frame', frame)

        # Pause/unpause video when 'p' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            paused = not paused
            if paused:
                pause_frame = frame
            else:
                pause_frame = None
        elif key == ord('q'):
            break

        # If video is paused, show current frame with grid and pixel count
        if paused:
            plt.cla()
            ax.imshow(frame)
            plt.xticks(np.arange(0, w, grid_size))
            plt.yticks(np.arange(0, h, grid_size))
            plt.grid(linewidth=0.5)
            for i in range(num_rows):
                for j in range(num_cols):
                    plt.text(j * grid_size, i * grid_size, str(grid[i, j]),
                             ha='center', va='center', fontsize=8)
            plt.pause(0.001)
        else:
            # Calculate number of pixels in current frame and print it
            num_pixels = np.prod(frame.shape[:2])
            print(f'Number of pixels in frame {frame_num}: {num_pixels}')

cap.release()

# Closes all the frames
cv2.destroyAllWindows()
plt.show()
