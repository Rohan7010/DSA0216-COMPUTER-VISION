#14. Transformation using Homography Matrix 
import cv2 
import numpy as np 
cap.release() 
cv2.destroyAllWindows() 
# Use ORB to detect keypoints 
img1 = cv2.imread('img1.jpg', 0) 
img2 = cv2.imread('img2.jpg', 0) 
orb = cv2.ORB_create() 
kp1, des1 = orb.detectAndCompute(img1, None) 
kp2, des2 = orb.detectAndCompute(img2, None) 
# Match features 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
matches = bf.match(des1, des2) 
matches = sorted(matches, key=lambda x: x.distance) 
# Use first 10 good matches 
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2) 
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2) 
# Get homography 
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC) 
warped = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0])) 
cv2.imwrite('homography_output.jpg', warped)
