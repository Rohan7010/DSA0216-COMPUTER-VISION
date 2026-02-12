# 3. Read an image in Python and show outline using Canny function

import cv2

img = cv2.imread("image.jpg")
edges = cv2.Canny(img, 100, 200)

cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
