#15. Transformation using Direct Linear Transformation (DLT) 
import cv2 
import numpy as np 
 
# Step 1: Read the image 
img = cv2.imread('E://opencv//t and t//456d1b45951425.584281dfe6460.jpg') 
if img is None: 
    print("Image not found. Make sure 'input.jpg' is in the same folder.") 
    exit() 
# Step 2: Define 4 points in the original image 
src_pts = np.float32([ 
    [100, 100], 
    [300, 100], 
    [300, 300], 
    [100, 300] 
]) 
# Step 3: Define where those points should map to in the output image 
dst_pts = np.float32([ 
    [80, 150], 
    [320, 100], 
    [310, 320], 
    [90, 330] 
]) 
# Step 4: Calculate the Homography matrix using DLT method 
H, status = cv2.findHomography(src_pts, dst_pts, method=0)  # method=0 means pure DLT 
print("Homography Matrix (DLT):\n", H) 
# Step 5: Apply the perspective transformation 
warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0])) 
# Step 6: Show the result 
cv2.imshow("Original Image", img) 
cv2.imshow("Warped Image using DLT", warped) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
