# 9. Perform rotation of an image to clockwise and counter-clockwise direction

import cv2

img = cv2.imread("image.jpg")
(h, w) = img.shape[:2]

M1 = cv2.getRotationMatrix2D((w/2, h/2), -90, 1)  # Clockwise
M2 = cv2.getRotationMatrix2D((w/2, h/2), 90, 1)   # Counter-clockwise

rotated_cw = cv2.warpAffine(img, M1, (w, h))
rotated_ccw = cv2.warpAffine(img, M2, (w, h))

cv2.imshow("Clockwise", rotated_cw)
cv2.imshow("Counter Clockwise", rotated_ccw)
cv2.waitKey(0)
cv2.destroyAllWindows()
