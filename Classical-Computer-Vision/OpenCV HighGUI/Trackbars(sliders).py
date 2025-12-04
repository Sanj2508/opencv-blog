import cv2
import os

# dummy callback function for the trackbar
def nothing(x):
    pass

image_path = "wild_animals.jpg"

# Check if the image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at {image_path}")

# Load the image once
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Failed to read the image. Check file integrity.")

# Create a named window
cv2.namedWindow('Trackbar Window')

# Create a trackbar
cv2.createTrackbar('Threshold', 'Trackbar Window', 0, 255, nothing)

while True:
    # Get current trackbar position
    threshold_value = cv2.getTrackbarPos('Threshold', 'Trackbar Window')

    # Apply threshold to the image
    _, thresh_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    # Show the result
    cv2.imshow('Trackbar Window', thresh_img)

    # Break loop on 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
