import cv2
import pathlib
import os
from os.path import exists

class FeaturesHaarDetection:
    '''Classificadores retirados de: https://github.com/opencv/opencv/tree/master/data/haarcascades'''

    def __init__(self):
        classifiers_base_path = str(pathlib.Path(__file__).parent.resolve()) + '\\classifiers\\'
        '''Resolvendo caminho dos diretórios'''
        viola_jones = exists(os.path.join(classifiers_base_path, 'haarcascade_frontalface_default.xml'))
        if not viola_jones:
            print('O caminho informado para o detector de face é inválido')
            print(os.path.join(classifiers_base_path, 'haarcascade_frontalface_default.xml'))
            raise FileNotFoundError
        self.face_detector = cv2.CascadeClassifier(os.path.join(classifiers_base_path, 'haarcascade_frontalface_default.xml'))

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.01, 3)
        for (x, y, w, h) in faces:
            print('Tem face')
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if faces == () :
            print('Não Tem face')
