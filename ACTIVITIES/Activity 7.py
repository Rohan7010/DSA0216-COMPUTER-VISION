import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Load Image in Grayscale
# -----------------------------
img = cv2.imread(r"C:\Users\rohan\Desktop\park_scene.jpg", 0)

if img is None:
    raise ValueError("Image not found. Check file path.")

print("Original Shape:", img.shape)

# -----------------------------
# 2️⃣ Generate Multiple Zoom Levels
# -----------------------------
images = [img]

for i in range(4):     # Create 4 lower resolutions
    images.append(cv2.pyrDown(images[-1]))

# -----------------------------
# 3️⃣ Compute Variance
# -----------------------------
variances = []

for level, im in enumerate(images):
    flat = im.flatten()
    var = np.var(flat)
    variances.append(var)
    print(f"Level {level} shape: {im.shape}, Variance: {var:.2f}")

# -----------------------------
# 4️⃣ Plot Zoom Level vs Variance
# -----------------------------
zoom_levels = list(range(len(images)))

plt.figure()
plt.plot(zoom_levels, variances)
plt.xlabel("Zoom Level")
plt.ylabel("Pixel Intensity Variance")
plt.title("Zoom Level vs Variance")
plt.show()
