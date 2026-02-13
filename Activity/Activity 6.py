import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread(r"C:\Users\rohan\Desktop\book_close.jpg")
img2 = cv2.imread(r"C:\Users\rohan\Desktop\book_far.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

print("Good matches:", len(good))

matched_img = cv2.drawMatches(gray1, kp1,
                              gray2, kp2,
                              good, None,
                              flags=2)

plt.imshow(matched_img)
plt.axis('off')
plt.show()
