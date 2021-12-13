from typing import List
import cv2
import pathlib
import os

from modules.person_detection.box import Box, Point

class PersonDetection:

    def __init__(self):
        self.class_names = []
        yolov4_files_base_path = str(pathlib.Path(__file__).parent.resolve()) + '\\yolov4\\'

        '''Resolvendo caminho dos diretórios'''
        coco_names_path = os.path.join(yolov4_files_base_path, 'coco.names')
        yolov4_w_path = os.path.join(yolov4_files_base_path, 'yolov4.weights')
        yolov4_cfg_path = os.path.join(yolov4_files_base_path, 'yolov4.cfg')

        with open(coco_names_path, 'r') as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

        net = cv2.dnn.readNet(yolov4_w_path ,yolov4_cfg_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    def detect_people(self, image, show: bool = False):
        '''Recebe um imagem, detecta as pessoas na cena e retorna, para cada pessoa encontrada,
        uma caixa delimitadora da região onde a pessoa se encontra. As coordenadas da caixa estão normalizadas'''
        CONFIDENCE_THRESHOLD = 0.2
        NMS_THRESHOLD = 0.4
        COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        
        classes, scores, boxes = self.model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        height_image, width_image, _ = image.shape
        
        people = []

        for (classid, score, box) in zip(classes, scores, boxes):

            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (self.class_names[classid[0]], score)
            if self.class_names[classid[0]] == 'person':
                width_box_person = box[2]
                height_box_person = box[3]
                if (width_box_person >= width_image) * 0.2 or (height_box_person >= height_image * 0.5):
                    box_person = Box(
                            Point(
                                box[0]/width_image, box[1]/height_image
                            ), 
                            Point(
                                (box[0] + box[2])/width_image, (box[1] + box[3])/height_image
                            )
                        )
                    people.append(box_person)
                    if show:
                        cv2.rectangle(image, box, color, 2)
                        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        return people