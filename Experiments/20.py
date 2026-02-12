#20. Sharpening of Image using Laplacian Mask with Negative Center Coefficient

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Laplacian kernel with negative center
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Apply filter
sharpened = cv2.filter2D(img, -1, laplacian_kernel)

# Display
plt.imshow(sharpened, cmap='gray')
plt.title("Laplacian Sharpening (Negative Center)")
plt.axis('off')
plt.show()
