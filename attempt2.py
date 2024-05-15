import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "C:/Users/raque/Downloads/zdo2024/images/incision_couples/SA_20240221-102238_so4qzqo8o15i_incision_crop_0.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bilateral filtering to reduce noise while preserving edges
bilateral_filtered = cv2.bilateralFilter(gray_image, 9, 75, 75)

# Edge detection using the Canny algorithm
edges = cv2.Canny(bilateral_filtered, 30, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image for visualization
contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

# Show results
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
# plt.title('Contours Detected')

plt.show()

print(f"Number of stitches detected: {len(contours)}")
