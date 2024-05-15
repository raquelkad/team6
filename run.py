import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get the image path from user input
image_path = input("Enter the path to the image: ")

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is None:
    print("Error: Image not found or invalid image format.")
else:
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred_image, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50, min_theta=np.pi / 2 - np.pi / 6, max_theta=np.pi / 2 + np.pi / 6)

    # Initialize a counter for stitches
    num_stitches = 0

    # Plot the original image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    # Draw detected lines on the image
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Filter lines based on vertical orientation
            if (np.pi / 6) < theta < (5 * np.pi / 6):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Increment stitch counter
                num_stitches += 1

        # Print the number of stitches detected
        print(f"Number of stitches detected: {num_stitches}")

    plt.show()
