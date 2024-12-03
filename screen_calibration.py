import cv2
import numpy as np
import tkinter as tk

root = tk.Tk()
SCREEN_WIDTH = root.winfo_screenwidth() + 59
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy()

left = int(SCREEN_WIDTH * 0.2)
middle_width = int(SCREEN_WIDTH * 0.5)
right = int(SCREEN_WIDTH * 0.8)
top = int(SCREEN_HEIGHT * 0.2)
middle_height = int(SCREEN_HEIGHT * 0.5)
bottom = int(SCREEN_HEIGHT * 0.8)

# Calibration points (e.g., a 3x3 grid)
CALIBRATION_POINTS = [
    (left,          top),  # Top-left
    (middle_width,  top),  # Top-center
    (right,         top),  # Top-right
    (left,          middle_height),  # Middle-left
    (middle_width,  middle_height),  # Center
    (right,         middle_height),  # Middle-right
    (left,          bottom),  # Bottom-left
    (middle_width,  bottom),  # Bottom-center
    (right,         bottom),  # Bottom-right
]

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
color = (255, 255, 255)
thickness = 5

def show_calibration_point_full_screen(point):
    """Display a single calibration point on the screen in full screen mode."""
    # Create a black screen
    image = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    # adjust point to be centered based on the size of the text
    (text_width, text_height), baseline = cv2.getTextSize("+", font, font_scale, thickness)
    x = point[0] - int(text_width // 2)
    y = point[1] - int(text_height // 2)
    position = (x, y)
    # Draw the calibration point (+)
    cv2.putText(image, "+", position, font, font_scale, color, thickness)
    #cv2.circle(image, (int(point[0]), int(point[1])), 20, (0, 255, 0), -1)

    # Set up the OpenCV window in full-screen mode
    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Show the image
    cv2.imshow(window_name, image)
    cv2.waitKey(1000)  # Show the point for 1 second

def get_gaze_data():
    """
    Simulate getting gaze data from an eye tracker.
    Replace with actual eye tracker SDK/API code.
    """
    # Example: Simulated gaze coordinate
    return (np.random.randint(0, SCREEN_WIDTH), np.random.randint(0, SCREEN_HEIGHT))


def show_countdown(count):
    image = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    # text font data
    (text_width, text_height), baseline = cv2.getTextSize(count, font, font_scale, thickness)
    x = middle_width - int(text_width // 2)
    y = middle_height - int(text_height // 2)
    window_name = "Calibration"

    # display countdown number on screen
    cv2.putText(image, count, (x, y), font, font_scale, color, thickness)
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, image)
    cv2.waitKey(1000)


def show_points():
    countdown = ["3", "2", "1"]
    for count in countdown:
        show_countdown(count)
    calibration_data = []
    for point in CALIBRATION_POINTS:
        show_calibration_point_full_screen(point)
        # Collect multiple gaze samples for accuracy
        gaze_samples = []
        for _ in range(20):  # Collect 20 samples per point
            gaze = get_gaze_data()
            gaze_samples.append(gaze)

        # Average the gaze samples to reduce noise
        average_gaze = np.mean(gaze_samples, axis=0)
        calibration_data.append((point, tuple(average_gaze)))

# Calibration data now contains [(screen_point, gaze_point), ...]

#from sklearn.linear_model import LinearRegression
#
## Prepare data for regression
#screen_points = np.array([p[0] for p in calibration_data])
#gaze_points = np.array([p[1] for p in calibration_data])
#
## Fit a linear regression model
#model_x = LinearRegression().fit(gaze_points[:, 0].reshape(-1, 1), screen_points[:, 0])
#model_y = LinearRegression().fit(gaze_points[:, 1].reshape(-1, 1), screen_points[:, 1])
#
#def map_gaze_to_screen(gaze_point):
#    """Map gaze coordinates to screen coordinates using the calibration model."""
#    screen_x = model_x.predict([[gaze_point[0]]])[0]
#    screen_y = model_y.predict([[gaze_point[1]]])[0]
#    return (screen_x, screen_y)

def main():
    show_points()

if __name__ == "__main__":
    main()