# 40. Draw Rectangular shape and extract objects. Auto-detect face and mark rectangle

import cv2
import os

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the input image
image = cv2.imread("E://opencv//t and t//OIP.jpeg")  # Replace with your image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Create folder to save extracted faces
os.makedirs("extracted_faces", exist_ok=True)

# Loop through detected faces and draw rectangle & save cropped face
for i, (x, y, w, h) in enumerate(faces):
    # Draw rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract face region
    face_crop = image[y:y + h, x:x + w]

    # Save cropped face
    face_filename = f"extracted_faces/face_{i+1}.jpg"
    cv2.imwrite(face_filename, face_crop)
    print(f"Saved: {face_filename}")

# Show the image with rectangles
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
