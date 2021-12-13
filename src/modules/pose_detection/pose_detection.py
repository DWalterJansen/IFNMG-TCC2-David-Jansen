from typing import List
import cv2
import mediapipe as mp

from modules.pose_detection.pose_landmarks import Point, PoseLandmarks

class PoseDetection:
    
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
    
    def detect_pose_landmarks(self, image, show: bool = False):
        with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
            height_image, width_image, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)
            if results.pose_landmarks is not None:
                if show:
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                landmark = results.pose_landmarks.landmark
                body_points = self.mp_pose.PoseLandmark

                nose = Point(x = landmark[body_points.NOSE].x, y = landmark[body_points.NOSE].y)
                right_eye = Point( x = landmark[body_points.RIGHT_EYE].x, y = landmark[body_points.RIGHT_EYE].y)
                left_eye = Point( x = landmark[body_points.LEFT_EYE].x, y = landmark[body_points.LEFT_EYE].y)
                mouth_right = Point( x = landmark[body_points.MOUTH_RIGHT].x, y = landmark[body_points.MOUTH_RIGHT].y)
                mouth_left = Point( x = landmark[body_points.MOUTH_LEFT].x, y = landmark[body_points.MOUTH_LEFT].y)
                right_ear = Point( x = landmark[body_points.RIGHT_EAR].x, y = landmark[body_points.RIGHT_EAR].y)
                left_ear = Point( x = landmark[body_points.LEFT_EAR].x, y = landmark[body_points.LEFT_EAR].y)
                right_shoulder = Point( x =  landmark[body_points.RIGHT_SHOULDER].x, y = landmark[body_points.RIGHT_SHOULDER].y)
                left_shoulder = Point( x =  landmark[body_points.LEFT_SHOULDER].x, y = landmark[body_points.LEFT_SHOULDER].y)

                return PoseLandmarks(
                    nose = nose,
                    right_eye = right_eye,
                    left_eye = left_eye,
                    mouth_right = mouth_right,
                    mouth_left = mouth_left,
                    right_ear = right_ear,
                    left_ear = left_ear,
                    right_shoulder = right_shoulder,
                    left_shoulder = left_shoulder,
                    width_image = width_image,
                    height_image = height_image
                )
            else:
                return None

    def detect_pose_in_multiples_images(self, images: List):
        landmarks_pose = []
        for id, image in enumerate(images):
            pose_landmarks_detected = self.detect_pose_landmarks(image=image)
            landmarks_pose.append((id, pose_landmarks_detected))
        return landmarks_pose