 #29. Morphological operations based on OpenCV using Erosion technique.
 #30. Morphological operations based on OpenCV using Dilation technique.
 #31. Morphological operations based on OpenCV using Opening technique.
 #32. Morphological operations based on OpenCV using Closing technique.
 #33. Morphological operations based on OpenCV using Morphological Gradient technique.
 #34. Morphological operations based on OpenCV using Top hat technique.
 #35. Morphological operations based on OpenCV using Black hat technique.
import cv2
import numpy as np

img = cv2.imread('lena.jpg', 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)

erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

# Choose which to show
cv2.imshow('Morph Operation1', erosion)
cv2.imshow('Morph Operation2', dilation)
cv2.imshow('Morph Operation3', opening)
cv2.imshow('Morph Operation4', closing)
cv2.imshow('Morph Operation5', tophat)
cv2.imshow('Morph Operation6', blackhat)




cv2.waitKey(0)
cv2.destroyAllWindows()
