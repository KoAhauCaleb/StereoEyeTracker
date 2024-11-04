import cv2
import numpy

face_cascade = cv2.CascadeClassifier('../Models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../Models/haarcascade_eye.xml')

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_face_img(img):
    gray_picture = grayscale(img)
    face_coords = face_cascade.detectMultiScale(gray_picture, 1.3, 5)
    x, y, w, h = face_coords[0]
    face = img[y:y+h, x:x+w] # cut the face frame out
    return face, face_coords

def blob_process(img, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, 110, 255, cv2.THRESH_BINARY)
    # img = cv2.dilate(img, None, iterations=4) #2
    # img = cv2.erode(img, None, iterations=2) #1
    # img = cv2.medianBlur(img, 5) #3

    keypoints = detector.detect(img)
    return keypoints, img

img = cv2.imread("poor_light_test.jpg")
face, face_coords = get_face_img(img)
for (x,y,w,h) in face_coords:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)



cv2.imshow('my image',img)
gray_face = grayscale(face)

eyes = eye_cascade.detectMultiScale(gray_face)
cut_eyes = []
for cnt, (ex,ey,ew,eh) in enumerate(eyes):
    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)
    eye = face[ey:ey+eh, ex:ex+ew]
    cv2.imshow(f'eye_pre {cnt}', eye)
    keypoints, thresh_img = blob_process(eye, detector)
    cv2.imshow(f'eye_thresh {cnt}', thresh_img)
    eye_key = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(f'eye_key {cnt}', eye_key)
    cut_eyes.append(eye)




for cnt, eye in enumerate(cut_eyes):
    cv2.imshow(f'eye{cnt}', eye)
cv2.imshow('my image2',face)
cv2.waitKey(0)
cv2.destroyAllWindows()