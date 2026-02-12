36. Watch Recognition (General Object Detection)
import cv2

img = cv2.imread('watch.jpg')
watch_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Replace with custom if needed
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
objects = watch_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in objects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Detected Object', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
