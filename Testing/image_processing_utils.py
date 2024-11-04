import cv2
import numpy as np


def preprocess_image(image_path):
    # Load the image
    original_image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to remove noise
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    return original_image, blurred_image


def detect_iris(blurred_image):
    # Use Hough Circles to detect circles in the image
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=20, maxRadius=80)
    return circles


def extract_iris(original_image, circles):
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle in the output image
            cv2.circle(original_image, (x, y), r, (0, 255, 0), 4)
            # Extract the iris region
            mask = np.zeros_like(original_image)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
            iris = cv2.bitwise_and(original_image, mask)
            return iris
    return None


def main(image_path):
    original_image, blurred_image = preprocess_image(image_path)
    circles = detect_iris(blurred_image)
    iris = extract_iris(original_image, circles)
    if iris is not None:
        # Display the iris
        cv2.imshow("Iris", iris)
        cv2.waitKey(0)
    else:
        print("No iris detected")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = 'path_to_your_image.jpg'
    main(image_path)