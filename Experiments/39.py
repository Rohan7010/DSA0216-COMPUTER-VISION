#39. Vehicle Detection
import cv2

car_cascade = cv2.CascadeClassifier('cars.xml')  # You need to download this XML
video = cv2.VideoCapture('traffic.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow('Vehicle Detection', frame)
    if cv2.waitKey(10) == 27:
        break

video.release()
cv2.destroyAllWindows()
