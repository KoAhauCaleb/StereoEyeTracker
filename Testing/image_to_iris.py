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

model = "face_landmarker_v2_with_blendshapes.task"

RESULT = None

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

        return annotated_image

def run():
    base_options = python.BaseOptions(model_asset_path=model)

    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.IMAGE,
                                           # result_callback=callback_result,
                                           num_faces=1,
                                           output_face_blendshapes=False,
                                           )

    img = cv2.imread("poor_light_test.jpg", cv2.IMREAD_COLOR)
    s = img.shape
    print(s)
    r = s[0] / 540
    y = s[1] / r
    shape = (int(y), 540)
    cv2.imshow("frame", cv2.resize(img, shape))
    stime = time.time()
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        stime = time.time()
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result = landmarker.detect(rgb_frame)
        print(result)
        annotation = draw_landmarks_on_image(img, result)
        cv2.imshow("annotation", cv2.resize(annotation, shape))
        cv2.waitKey(1)


    # get eye positions
    right_xmin = None
    right_xmax = None
    right_ymin = None
    right_ymax = None
    for idx in mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS:
        print(idx)
        x = result.face_landmarks[0][idx[0]].x
        y = result.face_landmarks[0][idx[0]].y
        if right_xmax is None or x > right_xmax:
            right_xmax = x
        if right_xmin is None or x < right_xmin:
            right_xmin = x
        if right_ymax is None or y > right_ymax:
            right_ymax = y
        if right_ymin is None or y < right_ymin:
            right_ymin = y


    print(right_xmin)
    print(right_xmax)
    print(right_ymin)
    print(right_ymax)

    right_xmin = int(right_xmin * s[1])
    right_xmax = int(right_xmax * s[1])
    right_ymin = int(right_ymin * s[0])
    right_ymax = int(right_ymax * s[0])

    print(right_xmin)
    print(right_xmax)
    print(right_ymin)
    print(right_ymax)

    x_ratio = (right_xmax - right_xmin) * 2.5
    y_ratio = (right_ymax - right_ymin) * 2.5

    eye_crop_size = (int(x_ratio), int(y_ratio))

    for idx in mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS:
        pass

    cropped = img[right_ymin:right_ymax, right_xmin:right_xmax]
    cropped = cv2.resize(cropped, eye_crop_size)
    cv2.imshow("cropped", cv2.resize(cropped, eye_crop_size))

    # get iris pixels
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    crop_shape = gray.shape
    center_thresh = gray[crop_shape[0]//2, crop_shape[1]//2]
    center_thresh = int(center_thresh * 1.1)
    print(f'{center_thresh=}')

    retval, thresholded = cv2.threshold(gray, center_thresh + 6, 255, cv2.THRESH_BINARY)
    # thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)
    cv2.imshow("threshold",  cv2.resize(thresholded, eye_crop_size))

    # cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    closed = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)
    cv2.imshow("closed", cv2.resize(closed, eye_crop_size))

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    drawing = np.copy(cropped)
    cv2.drawContours(drawing, contours, -1, (255, 0, 0), 2)

    accepted_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)
        bounding_box = cv2.boundingRect(contour)

        extend = area / (bounding_box[2] * bounding_box[3])

        if area < 5:
            continue

        circumference = cv2.arcLength(contour, True)
        circularity = circumference ** 2 / (4 * math.pi * area)

        if circularity > 3:
            continue

        # reject the contours with big extend
        if extend > 0.8:
            continue

        # calculate countour center and draw a dot there
        m = cv2.moments(contour)
        if m['m00'] != 0:
            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
            cv2.circle(drawing, center, 0, (255, 255, 0), -1)

        # fit an ellipse around the contour and draw it into the image
        try:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0))
        except:
            pass


        accepted_contours.append(contour)
        # accepted_contours.append(cv2.convexHull(contour))

    cv2.imshow("Drawing", cv2.resize(drawing, eye_crop_size))

    accepted = np.copy(cropped)

    biggest_contour = accepted_contours[0]
    for contour in accepted_contours:
        accepted_area = cv2.contourArea(biggest_contour)
        area = cv2.contourArea(contour)
        if accepted_area < area:
            biggest_contour = contour


    # cv2.drawContours(accepted, biggest_contour, -1, (255, 0, 0), 2)

    # cv2.waitKey(0)

    accepted = cv2.resize(accepted, eye_crop_size)

    ellipse = cv2.fitEllipse(biggest_contour)
    cv2.ellipse(accepted, box=ellipse, color=(0, 255, 0))
    print(ellipse)

    cv2.imshow("accepted", accepted)

    print(time.time() - stime)




def main():
    run()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
