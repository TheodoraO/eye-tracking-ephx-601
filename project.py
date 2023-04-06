import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage

# Load video file
vid = imageio.get_reader('EyeTracking.mp4')

# Set crop parameters for the video
x, y, w, h = 550, 58, 750, 260

# Calculate center of the eye region
eye_center_x = x + w // 2
eye_center_y = y + h // 2

# Set up figure to display video and pixel count
fig, ax = plt.subplots(figsize=(20, 15))
ax.set_xlim([0, w])
ax.set_ylim([0, h])
plt.xticks([])
plt.yticks([])
plt.grid(False)

# Loop through frames in video
for i, frame in enumerate(vid):
    # Crop frame to focus on monkey's face
    frame = frame[y:y + h, x:x + w]

    # Convert frame to grayscale
    gray = skimage.color.rgb2gray(frame)

    # Calculate number of pixels in the eye region
    eye_pixels = gray[eye_center_y - 30:eye_center_y + 30, eye_center_x - 30:eye_center_x + 30]
    num_pixels = int(np.sum(eye_pixels > 0.5))

    # Display frame with pixel count
    ax.imshow(frame[::-1, :], origin='lower', aspect='auto')  # Flip image vertically
    ax.text(10, 40, f'Frame {i + 1}: {num_pixels} pixels in eye', color='white', fontsize=24)
    plt.pause(0.01)

# Close the figure
plt.close(fig)
