# 2. Read an image in Python and convert the image to Blur using GaussianBlur

import cv2

img = cv2.imread("image.jpg")
blur = cv2.GaussianBlur(img, (15, 15), 0)

cv2.imshow("Blurred", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
