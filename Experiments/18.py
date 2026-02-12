#18. Edge Detection using Sobel Matrix along Y-axis
import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel Y
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Display
plt.imshow(sobel_y, cmap='gray')
plt.title("Sobel Edge Detection - Y axis")
plt.axis('off')
plt.show()
