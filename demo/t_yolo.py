from pytrt.yolo_infer import YoloModel
import cv2
labels = open('coco.names', 'r', encoding='utf-8').readlines()
labels = [l.strip('\n') for l in labels]

client = YoloModel('workspace/yolov8s.engine', "V8", labels)
img = client.single_inference('workspace/inference/gril.jpg')
imgs = client.batch_inference(['workspace/inference/car.jpg']*16)
img = client.single_inference('workspace/inference/gril.jpg')
cv2.imwrite('result.png', imgs[0])