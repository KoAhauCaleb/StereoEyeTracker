import time

import get_area
import cv2
import mediapipe
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

model = "face_landmarker.task"

RESULT = None

glints = get_area.GlintPoints(None, None)

frame_counter = 0
cached_glints_left = []
cached_glints_right = []

def draw_landmarks_on_image(rgb_image, detection_result):
    global frame_counter
    global cached_glints_left, cached_glints_right

    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    shape = rgb_image.shape

    landmarker = get_area.initialize_landmarker()

    frame_counter += 1

    if frame_counter % 2 == 0:
        # Detect glints for both eyes
        cached_glints_right = get_area.GlintPoints(
            get_area.run(shape, annotated_image, mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS, landmarker),
            None
        ).right_glint
        cached_glints_left = get_area.GlintPoints(
            None,
            get_area.run(shape, annotated_image, mp.solutions.face_mesh.FACEMESH_LEFT_IRIS, landmarker)
        ).left_glint

    # Draw cached glints on the image
    get_area.circle_image(annotated_image, cached_glints_right)
    get_area.circle_image(annotated_image, cached_glints_left)
    # Loop through the detected faces to visualize.
    #for idx in range(len(face_landmarks_list)):
    #    face_landmarks = face_landmarks_list[idx]
#
    #    # Draw the face landmarks.
    #    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #    face_landmarks_proto.landmark.extend([
    #        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    #    ])
#
    #    solutions.drawing_utils.draw_landmarks(
    #        image=annotated_image,
    #        landmark_list=face_landmarks_proto,
    #        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
    #        landmark_drawing_spec=None,
    #        connection_drawing_spec=mp.solutions.drawing_styles
    #        .get_default_face_mesh_tesselation_style())
    #    solutions.drawing_utils.draw_landmarks(
    #        image=annotated_image,
    #        landmark_list=face_landmarks_proto,
    #        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
    #        landmark_drawing_spec=None,
    #        connection_drawing_spec=mp.solutions.drawing_styles
    #        .get_default_face_mesh_contours_style())
    #    solutions.drawing_utils.draw_landmarks(
    #        image=annotated_image,
    #        landmark_list=face_landmarks_proto,
    #        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
    #        landmark_drawing_spec=None,
    #        connection_drawing_spec=mp.solutions.drawing_styles
    #        .get_default_face_mesh_iris_connections_style())
        
        #glint_L = get_area.run(shape, annotated_image, mp.solutions.face_mesh.FACEMESH_LEFT_EYE, "L")
        #glint_R = get_area.run(shape, annotated_image, mp.solutions.face_mesh.FACEMESH_RIGHT_EYE, "R")

    return annotated_image


def callback_result(result, frame, time):
    global RESULT
    try:
        annotated_image = draw_landmarks_on_image(frame.numpy_view(), result)
        RESULT = annotated_image
        print(result)
        # cv2.imshow('frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(e)

def run():
    base_options = python.BaseOptions(model_asset_path=model)

    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.IMAGE,
                                           # result_callback=callback_result,
                                           num_faces=1,
                                           output_face_blendshapes=False,
                                           )

    cap = cv2.VideoCapture(0)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect(rgb_frame)
            annotation = draw_landmarks_on_image(frame, result)
            cv2.imshow("frame", annotation)
            cv2.waitKey(1)



def run_async():
    global RESULT
    base_options = python.BaseOptions(model_asset_path=model)

    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.LIVE_STREAM,
                                           result_callback=callback_result,
                                           num_faces=1,
                                           )

    cap = cv2.VideoCapture(0)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        start = time.time() * 1000
        while cap.isOpened():
            ret, frame = cap.read()
            rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # print(time.time() * 1000 - start)
            landmarker.detect_async(rgb_frame, int(time.time() * 1000 - start))
            # annotation = draw_landmarks_on_image(frame, result)
            if RESULT is not None:
                cv2.imshow("frame", RESULT)
            cv2.waitKey(1)


def main():
    run_async()


if __name__ == "__main__":
    main()