from pytrt.yolo_infer import YoloModel
import cv2
labels = open('coco.names', 'r', encoding='utf-8').readlines()
labels = [l.strip('\n') for l in labels]

client = YoloModel('yolov8n.transd.engine', "V8", labels)
img = client.single_inference('inference/gril.jpg')
imgs = client.batch_inference(['inference/car.jpg']*16)
img = client.single_inference('inference/gril.jpg')
cv2.imwrite('result.png', imgs[0])