import cv2
import numpy
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = "hand_landmarker.task"),
    running_mode = VisionRunningMode.IMAGE,
    num_hands = 1
)

FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)



face_dectector = FaceLandmarker.create_from_options(face_options)
hand_dectector = HandLandmarker.create_from_options(hand_options)


cap = cv2.VideoCapture(0)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]


while True:

    success, img = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(mp.ImageFormat.SRGB, rgb)

    face_result = face_dectector.detect(mp_img)
    hand_result = hand_dectector.detect(mp_img)

    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            h, w, _ = img.shape
            lm_list = []


        for lm in hand:
            lm_list.append((int(lm.x *w),int(lm.y *h)))

        for x, y in lm_list:
            cv2.circle(img, (x, y), 5, (255, 0, 0), cv2.FILLED)

        for start, end in HAND_CONNECTIONS:
            cv2.line(img, lm_list[start], lm_list[end], (0, 255, 0), 2)

    cv2.imshow(img, "Eye color Change")

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()