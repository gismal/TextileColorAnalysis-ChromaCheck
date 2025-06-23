import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(gray_image)

# Apply bilateral filtering to reduce noise while keeping edges sharp
bilateral_filtered_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)

# Apply Gaussian blur to further smooth the image
blurred_image = cv2.GaussianBlur(bilateral_filtered_image, (5, 5), 0)

# Apply Sobel edge detection
sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = cv2.magnitude(sobelx, sobely)

# Convert Sobel edges to uint8
sobel_edges = cv2.convertScaleAbs(sobel_edges)

# Apply a threshold to get binary images from the Sobel edges
_, sobel_binary = cv2.threshold(sobel_edges, 100, 255, cv2.THRESH_BINARY)

# Apply morphological operations (opening and closing) to clean up the edges
kernel = np.ones((3, 3), np.uint8)
morphed_edges = cv2.morphologyEx(sobel_binary, cv2.MORPH_CLOSE, kernel)
morphed_edges = cv2.morphologyEx(morphed_edges, cv2.MORPH_OPEN, kernel)

# Find contours on the combined edge detected and thresholded image
contours, hierarchy = cv2.findContours(morphed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area and hierarchy, and exclude those touching the image borders
filtered_contours = [
    cnt for i, cnt in enumerate(contours)
    if cv2.contourArea(cnt) > 2000 and hierarchy[0, i, 3] == -1  # Increased area threshold
]

# Smooth the contours using approxPolyDP with a larger value
smoothed_contours = [cv2.approxPolyDP(cnt, 10, True) for cnt in filtered_contours]  # Increased epsilon value

# Draw the smoothed contours on the original image
contour_image = image.copy()
cv2.drawContours(contour_image, smoothed_contours, -1, (0, 0, 255), 2)  # Changed to red color for better visibility

# Display the original and contoured image using matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Refined Contoured Image with Sobel')
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))

plt.show()