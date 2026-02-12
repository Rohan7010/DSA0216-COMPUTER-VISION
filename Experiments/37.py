#37. Play Video in Reverse
import cv2

cap = cv2.VideoCapture('video.mp4')
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

for frame in reversed(frames):
    cv2.imshow('Reverse Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
