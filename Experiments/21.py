#21. Sharpening with Laplacian Mask (Diagonal Extension)


import cv2
import numpy as np

img = cv2.imread('image.jpg', 0)

laplacian_diagonal = np.array([[1, 1, 1],
                                [1, -8, 1],
                                [1, 1, 1]])

sharpened = cv2.filter2D(img, -1, laplacian_diagonal)

cv2.imshow('Sharpened Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
