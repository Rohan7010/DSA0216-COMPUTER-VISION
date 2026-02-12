# 17. Edge Detection using Sobel Matrix along X-axis

import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel X
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)

# Display
plt.imshow(sobel_x, cmap='gray')
plt.title("Sobel Edge Detection - X axis")
plt.axis('off')
plt.show()
