cmake_minimum_required(VERSION 3.16)
PROJECT(infer VERSION 1.0.0 LANGUAGES C CXX CUDA)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(CUDA_GEN_CODE "-gencode=arch=compute_72,code=sm_72")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(OpenCV_DIR "/usr/include/opencv4/")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4")
set(CUDNN_DIR "/usr/include/aarch64-linux-gnu")
set(TENSORRT_DIR "/usr/include/aarch64-linux-gnu")
set(PythonRoot "/sdk/envs/py3.8")
set(PythonName "python3.8")

find_package(CUDA REQUIRED)
find_package(OpenCV)

include_directories(
        /usr/include/${PythonName}
        ${PROJECT_SOURCE_DIR}/pytrt
        ${OpenCV_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        ${TENSORRT_DIR}/include
        ${TENSORRT_DIR}
        ${CUDNN_DIR}/include
        ${CUDNN_DIR}
        third_party/pybind11/include
)

link_directories(
        ${PythonRoot}/lib
        ${TENSORRT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib
        ${CUDNN_DIR}
        /usr/lib/aarch64-linux-gnu
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/pytrt/_lib/YoloModel/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/pytrt/_lib/YoloModel/*.cu)

add_subdirectory(pytrt/_lib/YoloModel)

#cuda_add_library(plugin_list SHARED ${cuda_srcs})
#target_link_libraries(plugin_list nvinfer nvinfer_plugin nvonnxparser)
#target_link_libraries(plugin_list cuda cublas cudart cudnn)
#target_link_libraries(plugin_list stdc++ dl)
#target_link_libraries(plugin_list ${OpenCV_LIBS})
#add_executable(infer ${cpp_srcs})
#target_link_libraries(infer nvinfer nvinfer_plugin nvonnxparser)
#target_link_libraries(infer cuda cublas cudart cudnn)
#target_link_libraries(infer stdc++ dl)
#target_link_libraries(infer plugin_list)
#target_link_libraries(infer ${OpenCV_LIBS})
