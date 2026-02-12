#22. Laplacian Mask with Positive Center Coefficient


import cv2
import numpy as np

img = cv2.imread('image.jpg', 0)

laplacian_positive = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])

sharpened = cv2.filter2D(img, -1, laplacian_positive)

cv2.imshow('Positive Laplacian Sharpened', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
