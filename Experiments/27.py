import cv2

img = cv2.imread('image.jpg')
crop = img[50:150, 50:150]
img[200:300, 200:300] = crop

cv2.imshow('Copy-Paste Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
