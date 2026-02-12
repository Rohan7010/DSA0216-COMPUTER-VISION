#high boost filtering 

import cv2
import numpy as np

img = cv2.imread('image.jpg', 0)
blur = cv2.GaussianBlur(img, (5,5), 0)
mask = cv2.subtract(img, blur)
k = 1.5
highboost = cv2.add(img, k * mask.astype(np.uint8))

cv2.imshow("High-Boost Filter", highboost)
cv2.waitKey(0)
cv2.destroyAllWindows()
