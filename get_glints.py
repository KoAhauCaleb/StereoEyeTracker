import math
import time

import cv2
import mediapipe
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from pycparser.ply.yacc import restart

model = "face_landmarker.task"

class GlintPoints:
    def __init__(self, right_glint, left_glint):
        self.right_glint = right_glint
        self.left_glint = left_glint

def initialize_landmarker():
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)

def get_eye_boundaries(result, eye, s):
    """ Get bounding box for the eye landmarks """
    xmin, xmax, ymin, ymax = None, None, None, None
    for idx in eye:
        x = result.face_landmarks[0][idx[0]].x
        y = result.face_landmarks[0][idx[0]].y
        if xmax is None or x > xmax:
            xmax = x
        if xmin is None or x < xmin:
            xmin = x
        if ymax is None or y > ymax:
            ymax = y
        if ymin is None or y < ymin:
            ymin = y
    xmax = int(xmax * s[1])
    xmin = int(xmin * s[1])
    ymax = int(ymax * s[0])
    ymin = int(ymin * s[0])
    width, height = (xmax-xmin), (ymax-ymin)
    width_padding, height_padding = int(width / 10), int(height / 10)
    return int(xmin + width_padding), int(xmax - width_padding), int(ymin  + height_padding), int(ymax - height_padding)

def detect_glints(cropped, cropped_area, cropped_width):#, eye_name, im_num):
    """ Detect glints in the cropped eye region """
    flat = cropped.flatten()
    min_threshold = int(np.percentile(flat, 98.5))
    _, threshold_image = cv2.threshold(cropped, min_threshold, 255, cv2.THRESH_BINARY)

    #cv2.imshow(f"Eye_{eye_name}_{im_num}", threshold_image)

    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(cropped_area / 600)
    max_area = int(cropped_area / 50)
    glints = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if min_area <= contour_area <= max_area:
            (x, y) = cv2.minEnclosingCircle(contour)[0] 
            radius = int(cv2.minEnclosingCircle(contour)[1])
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * math.pi * contour_area / (perimeter ** 2)
            if circularity < 0.9:
                glints.append(((x, y), radius))
    return glints

def run(s, img, eye, landmarker):#, eye_name, im_num):
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    result = landmarker.detect(rgb_frame)
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xmin, xmax, ymin, ymax = get_eye_boundaries(result, eye, s)
    cropped = grayscale_image[ymin:ymax, xmin:xmax]
    cropped_width = xmax - xmin
    cropped_height = ymax - ymin
    cropped_area = int((cropped_height) * (cropped_width))
    glints = detect_glints(cropped, cropped_area, cropped_width)#, eye_name, im_num)
    uncropped_glints = [((int(xmin + glint[0][0]), int(ymin + glint[0][1])), glint[1]) for glint in glints]

    return uncropped_glints

def circle_image(img, glints):
    for glint in glints:
        cv2.circle(img, glint[0], glint[1], (0, 255, 0), -1)
    return img

def main():
    img1 = cv2.imread("poor_light_test.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("more_testing.jpg", cv2.IMREAD_COLOR)
    img3 = cv2.imread("close.jpg", cv2.IMREAD_COLOR)
    img4 = cv2.imread("closer.jpg", cv2.IMREAD_COLOR)
    img5 = cv2.imread("closest.jpg", cv2.IMREAD_COLOR)
    s1 = img1.shape
    s2 = img2.shape
    s3 = img3.shape
    s4 = img4.shape
    s5 = img5.shape
    right_eye = mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS
    left_eye = mp.solutions.face_mesh.FACEMESH_LEFT_IRIS

    landmarker = initialize_landmarker()  # Load model only once

    left_glints_1 = run(s1, img1, left_eye, landmarker, "left", "1")
    right_glints_1 = run(s1, img1, right_eye, landmarker, "right", "1")
    circle_image(img1, left_glints_1)
    circle_image(img1, right_glints_1)

    left_glints_2 = run(s2, img2, left_eye, landmarker, "left", "2")
    right_glints_2 = run(s2, img2, right_eye, landmarker, "right", "2")
    circle_image(img2, left_glints_2)
    circle_image(img2, right_glints_2)

    left_glints_3 = run(s3, img3, left_eye, landmarker, "left", "3")
    right_glints_3 = run(s3, img3, right_eye, landmarker, "right", "3")
    circle_image(img3, left_glints_3)
    circle_image(img3, right_glints_3)

    left_glints_4 = run(s4, img4, left_eye, landmarker, "left", "4")
    right_glints_4 = run(s4, img4, right_eye, landmarker, "right", "4")
    circle_image(img4, left_glints_4)
    circle_image(img4, right_glints_4)

    left_glints_5 = run(s5, img5, left_eye, landmarker, "left", "5")
    right_glints_5 = run(s5, img5, right_eye, landmarker, "right", "5")
    circle_image(img5, left_glints_5)
    circle_image(img5, right_glints_5)

    cv2.imshow("Image_1", img1)
    cv2.imshow("Image_2", img2)
    cv2.imshow("Image_3", img3)
    cv2.imshow("Image_4", img4)
    cv2.imshow("Image_5", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()