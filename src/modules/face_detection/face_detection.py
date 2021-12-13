import math
import mediapipe as mp
import cv2

class FaceDetection:

    def __init__(self, min_detection_confidence=0.6):
        self.__mp_face_detector = mp.solutions.face_detection
        self.__face_detection = self.__mp_face_detector.FaceDetection(min_detection_confidence=0.7)

    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int, image_height: int):
            '''Normaliza as coordenadas do pixel'''
            # Verifica se o valor normalizado está entre 0 e 1.
            def is_valid_normalized_value(value: float):
                return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

            if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
                return None

            x_px = min(math.floor(normalized_x * image_width), image_width - 1)
            y_px = min(math.floor(normalized_y * image_height), image_height - 1)
            return x_px, y_px

    def face_limits(self, image, detection):
        if not detection.location_data:
            return

        image_rows, image_cols, _ = image.shape
        location = detection.location_data

        # Desenha o contorno se ele existe
        if not location.HasField('relative_bounding_box'):
            return
        relative_bounding_box = location.relative_bounding_box
        rect_start_point = self._normalized_to_pixel_coordinates(
            relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
            image_rows)
        rect_end_point = self._normalized_to_pixel_coordinates(
            relative_bounding_box.xmin + relative_bounding_box.width,
            relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
            image_rows)

        return dict({'start': rect_start_point, 'end': rect_end_point})

    def keypoints(self, image, detection):
        '''0 = olho esquerdo, 1 = olho direito, 2 = nariz, 3 = boca'''
        if not detection.location_data:
            return

        image_rows, image_cols, _ = image.shape
        location = detection.location_data

        keypoints = []
        for keypoint in location.relative_keypoints:
            keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                image_cols, image_rows)
            keypoints.append(keypoint_px)
        
        # 0 = olho esquerdo
        # 1 = olho direito
        # 2 = nariz
        # 3 = boca
        return keypoints[0:4]

    def process(self, image):
        height_image, width_image, _ = image.shape
        if height_image <= 0 or width_image <= 0:
            print('Size < 0')
            raise Exception
        results = self.__face_detection.process(image)
        return results.detections
