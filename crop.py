import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# Load the eye image

img = Image.open("eye_image.png")

# Convert the image to a numpy array
img_array = np.array(img)
# Display the image
plt.imshow(img_array)
plt.show()