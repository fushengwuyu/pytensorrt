//
// Created by sunshine on 24-2-2.
//

#include "objectdetect.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "yolo.hpp"

namespace py = pybind11;



PYBIND11_MODULE(libYOLODetector, m) {
    py::class_<YOLODetector,std::shared_ptr<YOLODetector>>(m, "YOLODetector")
    .def(py::init<const std::string&, yolo::Type, const std::vector<std::string>&, float, float>(),
            py::arg("modelPath"), py::arg("modelType"), py::arg("labels"),py::arg("confidence_threshold"), py::arg("nms_threshold"))
    .def("singleInference", &YOLODetector::singleInference,py::arg("imagePath"), py::arg("names"))
    .def("batchInference", &YOLODetector::batchInference,py::arg("imagePaths"), py::arg("names"));

    py::enum_<yolo::Type>(m, "YoloType")
    .value("V5", yolo::Type::V5)
    .value("X", yolo::Type::X)
    .value("V3", yolo::Type::V3)
    .value("V7", yolo::Type::V7)
    .value("V8", yolo::Type::V8)
    .value("V8Seg", yolo::Type::V8Seg)
    .export_values();

    py::class_<cv::Mat>(m, "Image", py::buffer_protocol())
    .def_buffer([](cv::Mat& im) -> py::buffer_info {
    return py::buffer_info(
            // Pointer to buffer
            im.data,
    // Size of one scalar
    sizeof(unsigned char),
    // Python struct-style format descriptor
    py::format_descriptor<unsigned char>::format(),
    // Number of dimensions
    3,
    // Buffer dimensions
    {im.rows, im.cols, im.channels()},
    // Strides (in bytes) for each index
    {
    sizeof(unsigned char) * im.channels() * im.cols,
    sizeof(unsigned char) * im.channels(),
    sizeof(unsigned char)
    }
    );
    });

}
