import cv2

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("config/coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

image = cv2.imread('<image>.jpg')

net = cv2.dnn.readNet("config/yolov4-tiny.weights", "config/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    if class_names[classid[0]] == 'person':
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Detections", image)
cv2.destroyAllWindows()