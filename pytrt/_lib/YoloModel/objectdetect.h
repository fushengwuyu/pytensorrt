//
// Created by sunshine on 24-2-2.
//

#ifndef INFER_OBJECTDETECT_H
#define INFER_OBJECTDETECT_H
#include <opencv2/opencv.hpp>
#include <memory>
#include "yolo.hpp"

class YOLODetector {
public:
        YOLODetector(const std::string& modelPath, yolo::Type modelType, const std::vector<std::string>& labels, float confidence_threshold = 0.25f, float nms_threshold = 0.5f);
    cv::Mat singleInference(const std::string& imagePath, std::vector<std::string> names);
    std::vector<cv::Mat> batchInference(const std::vector<std::string>& imagePaths, std::vector<std::string> names);

private:
    std::string modelPath_;
    yolo::Type modelType_;
    std::vector<std::string> labels_;
    float confidence_threshold_;
    float nms_threshold_;
    std::shared_ptr<yolo::Infer> yoloModel_;
    static yolo::Image cvimg(const cv::Mat &image);
};


#endif //INFER_OBJECTDETECT_H
