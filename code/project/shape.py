import cv2
import numpy as np

# Read an image
image = cv2.imread('project/img/shape.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
ret, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
binary_image  = 255-binary_image
cv2.imshow('Binary Image', binary_image)
# Find contours
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the total area of the image
total_area = image.shape[0] * image.shape[1]

# Draw contours with random colors for areas less than 80% of the image area
contour_image = np.zeros_like(image)
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    print(area)
    if area < (0.8 * total_area):
        color = np.random.randint(0, 255, size=3).tolist()  # Generate a random color
        cv2.drawContours(contour_image, [contour], -1, color, thickness=cv2.FILLED)

# Display the original and filled contour images
cv2.imshow('Original Image', image)
cv2.imshow('Filled Contour Image', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
