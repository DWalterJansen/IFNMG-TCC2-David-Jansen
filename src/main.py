from modules.face_detection.face_detection import FaceDetection # Classe para detecção de face utilizando o mediapie
from modules.normalize_coords import * # Método para converter coordenadas normalizadas em valores inteiros de pixels
from modules.args import argsParser # Método para centralizar os argumentos de execução
from imutils.video import WebcamVideoStream # Lib para capturar imagens da webcam com multithreading
from modules.execptions import * # Excetions customizadas
from os.path import exists # Método para verificar a existência de um arquivo
from modules.pose_detection.pose_landmarks import Point # Classe para representar pontos 2D
from modules.person_detection.person_detection import PersonDetection # Classe pars detecção de pessoas utilizando Yolo v4
from modules.pose_detection.pose_detection import PoseDetection # Classe para detecção de poses
import numpy as np # lib para processamento de arranjos e matrizes
import cv2 # opencv: construído com Nvidia Cuda e Cudnn
import os
from modules.skin_segmentation.skin_segmentation import SegmentationSkin

import time

def detect(image, show: bool = False):
    '''Detecta na imagem pessoas usando máscara:
        Estágios:
            - 1 - Detecta pessoas na cena e devolve suas regiões
            - 2 - Separamos a região da cabeça com base na região do corpo da pessoa
            - 4 - Buscamos encontrar uma face na região da cebeça
            - 4 - Caso seja obtida uma face no estágio anterior, é feita a segmentação de pele para verificar se parte
                    da face está oclusa por uma máscara
    '''

    width = int(image.shape[1])
    height = int(image.shape[0])
    # Estágio 1
    people = person_detection.detect_people(image=image, show=False)

    detections = []
    for id, person in enumerate(people):
        xi, yi = normalized_to_pixel_coordinates(person.p1.x, person.p1.y, width, height)
        xf, yf = normalized_to_pixel_coordinates(person.p2.x, person.p2.y, width, height)
        person_height, person_width = (yf-yi + 1, xf - xi + 1)

        # Estágio 2
        if person_width/person_height <= 0.3:
            yf = round(yi + (person_height*0.3))
            person_cropped = image[
                yi:yf,  # y
                xi:xf  # x
            ]
        elif person_width/person_height <= 0.5:
            yf = round(yi + (person_height*0.5))
            person_cropped = image[
                yi:yf,  # y
                xi:xf  # x
            ]
        elif person_width/person_height <= 0.8:
            yf = round(yi + (person_height*0.8))
            person_cropped = image[
                yi:yf,  # y
                xi:xf  # x
            ]
        else:
            person_cropped = image[
                yi:yf,  # y
                xi:xf  # x
            ]

        # Estágio 3
        person_cropped = cv2.cvtColor(person_cropped, cv2.COLOR_BGR2RGB)
        faces = face_detection.process(person_cropped)
        person_cropped = cv2.cvtColor(person_cropped, cv2.COLOR_RGB2BGR)

        if faces and len(faces) == 1 :
            face_limits = face_detection.face_limits(image=person_cropped, detection=faces[0])
            keypoints = face_detection.keypoints(image=person_cropped, detection=faces[0])

            # for id, keypoint in enumerate(keypoints):
            #     # Desenha os keypoints na face recortada da imagem original
            #     if keypoint is not None and keypoint[0] is not None and keypoint[1] is not None:
            #         center = (xi + keypoint[0], yi + keypoint[1])
            #         if id == 0:
            #             cv2.circle(image, center, 1, (0, 255, 255), 2)
            #         elif id == 1:
            #             cv2.circle(image, center, 1, (100, 100, 255), 2)
            #         else:
            #             cv2.circle(image, center, 1, (0, 255, 0), 2)
            if keypoints is not None and keypoints[0] is not None and keypoints[1] is not None and keypoints[2]:
                distx_nose_right_eye = keypoints[2][0] - keypoints[0][0]
                distx_nose_left_eye = keypoints[1][0] - keypoints[2][0]
                
                offset = round(person_width * 0.01)

                if face_limits['start'] is not None and face_limits['end'] is not None:
                    width_face = face_limits['end'][0] - face_limits['start'][0]
                    if distx_nose_right_eye > offset * distx_nose_left_eye:
                        xi_face = keypoints[0][0]
                        xf_face = int(face_limits['end'][0] - width_face * 0.1)
                    elif offset * distx_nose_right_eye < distx_nose_left_eye:
                        xi_face = int(face_limits['start'][0] + width_face * 0.1)
                        xf_face = keypoints[1][0]
                    else:
                        xi_face = int(face_limits['start'][0] + width_face * 0.1)
                        xf_face = int(face_limits['end'][0] - width_face * 0.1)

                    height_face = face_limits['end'][1] -  face_limits['start'][1]
                    yi_face = int(face_limits['start'][1] + height_face * 0.5)
                    yf_face = int(face_limits['end'][1])
                    
                    if yf_face - yi_face > 0 and xf_face - xi_face > 0: 
                        face_cropped = person_cropped[
                            yi_face:yf_face,  # y
                            xi_face:xf_face  # x
                        ]
                        # Estágio 4
                        segmentation_face_cropped = segmentation_skin.segmentation_bgr(face_cropped)
                        height_face, width_face, _ = face_cropped.shape
                        size_face = height_face * width_face
                        black_sample = [0,0,0]
                        black  = np.count_nonzero(np.all(segmentation_face_cropped==black_sample,axis=2))
                        
                        x_start_face = xi + int(face_limits['start'][0])
                        y_start_face = yi + int(face_limits['start'][1])

                        x_end_face = xi + int(face_limits['end'][0])
                        y_end_face = yi + int(face_limits['end'][1])

                        x_center_face = int((x_start_face + x_end_face)/2)
                        y_center_face = int((y_start_face + y_end_face)/2)
                        cv2.rectangle(image, (x_start_face, y_start_face), (x_end_face, y_end_face), (255, 255, 0), 1)
                        
                        if (black / size_face) >= 0.85 :
                            cv2.putText(image, f'Mask',  (x_start_face, y_start_face - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                            detections.append([x_center_face, y_center_face, 0])
                        else:
                            detections.append([x_center_face, y_center_face, 1])
                            cv2.putText(image, f'No Mask', (x_start_face, y_start_face - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
        
    cv2.imshow('Imagem', image)
    return detections
    

def teste(path: str):
    '''Realiza teste de detecção'''
    qtd_files = int(len(os.listdir(path))/3)
    right = 0
    wrong = 0
    not_detection = 0
    for i in range(qtd_files):
        image = cv2.imread(f'{path}\{i}.jpg')
        scale_percent = 30 # porcentagem da imagem original
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        new_size = (width, height)
        
        # Redimensiona a imagem
        resized = cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)
        detections = detect(image=resized)
        with open(f'{path}/{i}.txt') as f:
            lines = f.readlines()
            for line in lines:
                label = int(line.split(' ')[0])
                x_center = float(line.split(' ')[1])
                y_center = float(line.split(' ')[2])
                w_box = float(line.split(' ')[3])
                h_box = float(line.split(' ')[4])
                hit = False
                x_center, y_center = normalized_to_pixel_coordinates(x_center, y_center, width, height)
                w_box, h_box = normalized_to_pixel_coordinates(w_box, h_box, width, height)

                x_start_mark = int(x_center - w_box/2 + 1)
                x_end_mark = int(x_center + w_box/2 - 1)
                y_start_mark = int(y_center - h_box/2 + 1)
                y_end_mark = int(y_center + h_box/2 - 1)

                cv2.rectangle(resized, (x_start_mark, y_start_mark), (x_end_mark, y_end_mark), (255, 255, 0), 3)
                for detection in detections:
                    x_center_face = detection[0]
                    y_center_face = detection[1]
                    cv2.circle(resized, (x_center_face, y_center_face), 2, (0, 0, 255), 4)
                    result = detection[2]
                    if (x_center_face > x_start_mark and x_center_face < x_end_mark and y_center_face > y_start_mark and y_center_face < y_end_mark):
                        if result == label:
                            right = right + 1
                        else:
                            wrong = wrong + 1
                        hit = True

                if not hit:
                    not_detection = not_detection + 1

        #cv2.imshow('Imagem', resized)
        #cv2.waitKey()
    print(right)
    print(wrong)
    print(not_detection)



def face_masked_detection_video_cam(src:int = 0):
    '''Realiza a detecção com base na entrada da webcam'''

    vs = WebcamVideoStream(src=0).start()
    while True:
        image = vs.read()
        detect(image=image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()
    vs.stream.release()


def face_masked_detection_video_path(path:str):
    '''Realiza a detecção com base no caminho para um arquivo de vídeo'''
    
    video_exist = exists(path)
    if not video_exist:
        print('O caminho informado para o vídeo é inválido')
        raise VideoNotExist
    
    cap = cv2.VideoCapture(path)
    image_count = 0
    face_count = 0
    time_seconds = 0
    while cap.isOpened():
        start = time.perf_counter()
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            break

        scale_percent = 30 # porcentagem da imagem original
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        new_size = (width, height)
        
        # Redimensiona a imagem
        resized = cv2.resize(frame, new_size, interpolation = cv2.INTER_AREA)
        detections = detect(image=resized)
        if detections is not None:
            face_count = face_count + len(detections)
        finish = time.perf_counter()
        image_count = image_count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        time_seconds = time_seconds + finish - start
    print(f'Finished in {round(time_seconds, 2)} second(s). Frames: {image_count}. Faces: {face_count}')
    print(f'Time per image: {round(time_seconds, 2)/image_count}')
    print(f'Time per image: {round(time_seconds, 2)/face_count}')
    cv2.destroyAllWindows()


def face_masked_detection_image_path(path:str):
    '''Realiza a detecção com base no caminho de uma imagem'''

    image_exist = exists(path)
    if not image_exist:
        print('O caminho informado para a imagem é inválido')
        raise ImageNotExist

    image = cv2.imread(path)
    detect(image=image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def face_masked_detection(args):
    '''src: Fonte para detecção: 0: webcam, 1: vídeo, 2: imagem'''
    '''path: Caminho para o vídeo ou imagem'''

    if args['src'] == 0:
        face_masked_detection_video_cam()
    elif args['src'] == 1:
        face_masked_detection_video_path(path=args['path'])
    elif args['src'] == 2:
        face_masked_detection_image_path(path=args['path'])
    elif args['src'] == 3:
        teste(path=args['path'])


# Executa o módulo principal
if __name__ == '__main__':
    args = argsParser()
    person_detection = PersonDetection()
    pose_detection = PoseDetection()
    face_detection = FaceDetection()
    segmentation_skin = SegmentationSkin()
    face_masked_detection(args=args)
