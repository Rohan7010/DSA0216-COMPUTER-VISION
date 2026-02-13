import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create synthetic blobs
canvas = np.zeros((300,300), dtype=np.uint8)
cv2.circle(canvas, (75,150), 10, 255, -1)   # small blob
cv2.circle(canvas, (225,150), 40, 255, -1)  # large blob

# Apply Laplacian of Gaussian filters with different kernel sizes
log_small = cv2.GaussianBlur(canvas, (5,5), 0)
log_small = cv2.Laplacian(log_small, cv2.CV_64F)

log_large = cv2.GaussianBlur(canvas, (21,21), 0)
log_large = cv2.Laplacian(log_large, cv2.CV_64F)

# Display
plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(canvas, cmap='gray'); plt.title("Blobs")
plt.subplot(1,3,2); plt.imshow(log_small, cmap='gray'); plt.title("LoG Small Filter")
plt.subplot(1,3,3); plt.imshow(log_large, cmap='gray'); plt.title("LoG Large Filter")
plt.show()
