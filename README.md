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
cd pytrt
python setup.py install
```

### 4. 调用
在`demo`目录下有测试脚本`t_yolo.py`，执行
```shell
python t_yolo.py
```
即可查看调用结果
### 参考
https://github.com/shouxieai/infer  
https://github.com/shouxieai/tensorRT_Pro  
https://github.com/Melody-Zhou/tensorRT_Pro-YOLOv8
