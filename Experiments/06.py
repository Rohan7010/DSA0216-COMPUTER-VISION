# 6. Read a captured video in Python and display the video in slow and fast motion

import cv2

cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Video", frame)
    
    key = cv2.waitKey(50)  # Change value for fast (1), normal (50), slow (100)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
