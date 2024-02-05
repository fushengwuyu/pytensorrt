
aux_source_directory(. DIR_SRCS)
add_library(YOLODetector SHARED ${DIR_SRCS})
set_target_properties(YOLODetector PROPERTIES OUTPUT_NAME "YOLODetector")
set_target_properties(YOLODetector PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pytrt/_lib"
)
target_link_libraries(YOLODetector)
target_link_libraries(YOLODetector ${OpenCV_LIBS})
target_link_libraries(YOLODetector nvinfer nvinfer_plugin nvonnxparser)
target_link_libraries(YOLODetector cuda cublas cudart cudnn)
target_link_libraries(YOLODetector stdc++ dl)
