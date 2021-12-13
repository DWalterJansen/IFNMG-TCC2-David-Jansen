import numpy as np
import cv2

class SegmentationSkin:

    def __init__(self):
        # Lower and upper boundaries for the HSV skin segmentation method:
        self.lower_hsv = np.array([0, 48, 80], dtype="uint8")
        self.upper_hsv = np.array([20, 255, 255], dtype="uint8")

    # Skin detector based on the HSV color spaces
    def skin_detector_hsv(self, bgr_image):
        """Skin segmentation algorithm based on the HSV color space"""

        # Convert image from BGR to HSV color space:
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Find region with skin tone in HSV image:
        skin_region = cv2.inRange(hsv_image, self.lower_hsv, self.upper_hsv)
        return skin_region

    def segmentation_hsv(self, bgr_image): 
        '''bgr é o formato padrão do imread do opencv'''
        detected_skin = self.skin_detector_hsv(bgr_image)
        bgr = cv2.cvtColor(detected_skin, cv2.COLOR_GRAY2BGR)
        return bgr

    def bgr_skin(self, b, g, r):
        """Rule for skin pixel segmentation based on the paper 'RGB-H-CbCr Skin Colour Model for Human Face Detection'"""

        e1 = bool((r > 95) and (g > 40) and (b > 20) and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (
        abs(int(r) - int(g)) > 15) and (r > g) and (r > b))
        e2 = bool((r > 220) and (g > 210) and (b > 170) and (abs(int(r) - int(g)) <= 15) and (r > b) and (g > b))
        return e1 or e2

    # Skin detector based on the BGR color space
    def skin_detector_bgr(self, bgr_image):
        """Skin segmentation based on the RGB color space"""

        h = bgr_image.shape[0]
        w = bgr_image.shape[1]

        # We crete the result image with back background
        res = np.zeros((h, w, 1), dtype="uint8")

        # Only 'skin pixels' will be set to white (255) in the res image:
        for y in range(0, h):
            for x in range(0, w):
                (b, g, r) = bgr_image[y, x]
                if self.bgr_skin(b, g, r):
                    res[y, x] = 255

        return res

    def segmentation_bgr(self, bgr_image): 
        '''bgr é o formato padrão do imread do opencv'''
        detected_skin = self.skin_detector_bgr(bgr_image)
        bgr = cv2.cvtColor(detected_skin, cv2.COLOR_GRAY2BGR)
        return bgr