class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class PoseLandmarks:
    def __init__(self, nose: Point, right_eye: Point, left_eye: Point, mouth_right: Point, mouth_left: Point, right_ear: Point, left_ear: Point, right_shoulder: Point, left_shoulder: Point, width_image: int, height_image: int):
        self.nose = nose
        self.right_eye = right_eye
        self.left_eye = left_eye
        self.mouth_right = mouth_right
        self.mouth_left = mouth_left
        self.right_ear = right_ear
        self.left_ear = left_ear
        self.right_shoulder = right_shoulder
        self.left_shoulder = left_shoulder
        self.width_image = width_image
        self.height_image = height_image