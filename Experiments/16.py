#16. Edge Detection using Canny Method

import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
canny_edges = cv2.Canny(img, 100, 200)

# Display
plt.imshow(canny_edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()
