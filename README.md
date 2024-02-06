该仓库基于[infer](https://github.com/shouxieai/infer ) ，在原仓库的基础上，做了以下支持：
* 增加了pyton接口
* 使用cmake编译

### 1. 环境配置
在`CMakeLists.txt`文件中配置所需的依赖：
* set(CUDA_GEN_CODE "-gencode=arch=compute_72,code=sm_72")  
  配置框架算力值，我使用的是NVIDIA AGX XAVIER,默认是72，不同显卡算力可参考：https://zhuanlan.zhihu.com/p/579183464
* set(OpenCV_DIR "/usr/include/opencv4/")
* set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4")
* set(CUDNN_DIR "/usr/include/aarch64-linux-gnu")
* set(TENSORRT_DIR "/usr/include/aarch64-linux-gnu")
* set(PythonRoot "/sdk/envs/py3.8")
* set(PythonName "python3.8")

### 2. 编译
```shell
mkdir build 
cd build 
cmake ..
make -j6
```
编译成功后会在`pytrt/_lib`目录下面生成`libYOLODetector.so`动态库，python就是调用该动态库进行推理。

### 3. 安装
```shell
python setup.py install
```

### 4. 调用
在`demo`目录下有测试脚本`t_yolo.py`，执行
```shell
python t_yolo.py
```
即可查看调用结果

### 5. 性能

| 模型           | batch(1)             | batch(16)          |
| -------------- | -------------------- | ------------------ |
| yolov8s.engine | 0.04387497901916504  | 0.9142205715179443 |
| yolov8n.engine | 0.024980545043945312 | 0.6158907413482666 |

### 6. yolov8模型导出

1. 下载 YOLOv8

    ```shell
    git clone https://github.com/ultralytics/ultralytics.git
    ```

2. 修改代码, 保证动态 batch

    ```python
    # ========== head.py ==========

    # ultralytics/nn/modules/head.py第72行，forward函数
    # return y if self.export else (y, x)
    # 修改为：

    return y.permute(0, 2, 1) if self.export else (y, x)

    # ========== exporter.py ==========

    # ultralytics/engine/exporter.py第323行
    # output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
    # dynamic = self.args.dynamic
    # if dynamic:
    #     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
    #     if isinstance(self.model, SegmentationModel):
    #         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
    #         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    #     elif isinstance(self.model, DetectionModel):
    #         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
    # 修改为：

    output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output']
    dynamic = self.args.dynamic
    if dynamic:
        dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
        if isinstance(self.model, SegmentationModel):
            dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
            dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
        elif isinstance(self.model, DetectionModel):
            dynamic['output'] = {0: 'batch'}  # shape(1, 84, 8400)
    ```

3. 导出 onnx 模型, 在 ultralytics-main 新建导出文件 `export.py` 内容如下：

    ```python
    # ========== export.py ==========
    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")

    success = model.export(format="onnx", dynamic=True, simplify=True)
    ```
    
    执行`python export.py` 导出onnx模型。

4. 转换为tensorrt模型

   单张图片预测

   ```shell
   trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine
   ```

   批量图片预测

   ```shell
   trtexec --onnx=yolov8n.transd.onnx \
       --minShapes=images:1x3x640x640 \
       --maxShapes=images:16x3x640x640 \
       --optShapes=images:1x3x640x640 \
       --saveEngine=yolov8n.transd.engine
   ```

   

### 参考
https://github.com/shouxieai/infer  
https://github.com/shouxieai/tensorRT_Pro  
https://github.com/Melody-Zhou/tensorRT_Pro-YOLOv8