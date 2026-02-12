# 4. Read an image in Python and Dilate the image using cv2.dilate

import cv2
import numpy as np

img = cv2.imread("image.jpg", 0)
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(img, kernel, iterations=1)

cv2.imshow("Dilated", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
