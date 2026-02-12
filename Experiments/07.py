# 7. Capture video from webcam and display it in slow and fast motion

import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)  # Change to 100 for slow
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
