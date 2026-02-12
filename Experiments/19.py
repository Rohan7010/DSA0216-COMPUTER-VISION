#19. Edge Detection using Sobel Matrix along XY-axis
import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel X and Y
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Convert to absolute
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Combine
sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Display
plt.imshow(sobel_xy, cmap='gray')
plt.title("Sobel Edge Detection - XY axis")
plt.axis('off')
plt.show()
