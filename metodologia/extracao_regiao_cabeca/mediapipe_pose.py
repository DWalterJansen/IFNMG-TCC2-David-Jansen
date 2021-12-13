import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5) as pose:

    image = cv2.imread("images/<person>.jpg")
    height, width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image)
    print("Pose landmarks", results.pose_landmarks)
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()