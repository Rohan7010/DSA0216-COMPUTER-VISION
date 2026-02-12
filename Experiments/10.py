# 10. Perform moving of an image from one place to another

import cv2
import numpy as np

img = cv2.imread("image.jpg")
rows, cols = img.shape[:2]

M = np.float32([[1, 0, 100], [0, 1, 50]])  # Translate image
shifted = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow("Shifted Image", shifted)
cv2.waitKey(0)
cv2.destroyAllWindows()
