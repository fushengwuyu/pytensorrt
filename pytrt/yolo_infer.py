from pytrt._lib.libYOLODetector import YOLODetector, YoloType
from typing import List
import numpy as np
import time


def get_time(func):
    def inner(*arg, **kwargs):
        s_time = time.time()
        res = func(*arg, **kwargs)
        e_time = time.time()
        print('函数 {} 执行耗时: {} 秒'.format(func.__name__, e_time - s_time))
        return res

    return inner


type_mapping = dict(
    V5=YoloType.V5,
    V3=YoloType.V3,
    V7=YoloType.V7,
    V8=YoloType.V8,
    V8Seg=YoloType.V8Seg
)


class YoloModel:
    def __init__(self, engine_path, yolo_type="v8", labels=None, confidence_threshold=0.25, nms_threshold=0.5):
        self.labels = labels
        self.model = YOLODetector(engine_path, type_mapping.get(yolo_type), labels, confidence_threshold, nms_threshold)

    @get_time
    def single_inference(self, image_path: str, names=None):
        if names is None:
            names = self.labels
        img = self.model.singleInference(image_path, names)
        return np.array(img, copy=False)

    @get_time
    def batch_inference(self, image_paths: List[str], names=None):
        if names is None:
            names = self.labels
        img_list = self.model.batchInference(image_paths, names)
        return [np.array(img, copy=False) for img in img_list]
