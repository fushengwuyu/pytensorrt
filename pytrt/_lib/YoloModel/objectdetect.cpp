//
// Created by sunshine on 24-2-2.
//

#include "objectdetect.h"
#include <opencv2/opencv.hpp>

#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"
using namespace std;


//YOLODetector::YOLODetector(const std::string& modelPath, yolo::Type modelType, const std::vector<std::string>& labels, float confidence_threshold, float nms_threshold)
//        : modelPath_(modelPath), modelType_(modelType), labels_(labels), confidence_threshold(confidence_threshold), nms_threshold(nms_threshold){
//    yoloModel_ = yolo::load(modelPath_.c_str(), modelType_, confidence_threshold, nms_threshold);
//}
YOLODetector::YOLODetector(const std::string& modelPath, yolo::Type modelType, const std::vector<std::string>& labels, float confidence_threshold, float nms_threshold)
        : modelPath_(modelPath), modelType_(modelType), labels_(labels), confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold) {
    yoloModel_ = yolo::load(modelPath_.c_str(), modelType_, confidence_threshold_, nms_threshold_);
}

yolo::Image YOLODetector::cvimg(const cv::Mat &image){
    return yolo::Image(image.data, image.cols, image.rows);
}


cv::Mat YOLODetector::singleInference(const std::string& imagePath, std::vector<std::string> names) {
    cv::Mat image = cv::imread(imagePath);
    if (!yoloModel_) {
        std::cerr << "Failed to load YOLO model." << std::endl;
        return image;
    }
    auto yoloImage = YOLODetector::cvimg(image);
    auto objs = yoloModel_->forward(yoloImage);
    int i = 0;
    for (auto& obj : objs) {
        uint8_t b, g, r;
        std::tie(b, g, r) = yolo::random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                      cv::Scalar(b, g, r), 5);

        auto name = labels_[obj.class_label];
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            continue;
        }
        auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                      cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    return image;
}

std::vector<cv::Mat> YOLODetector::batchInference(const std::vector<std::string>& imagePaths, std::vector<std::string> names) {
    std::vector<cv::Mat> images;

    // 从文件路径读取图片并添加到向量中
    for (const auto& img_path : imagePaths) {
        cv::Mat img = cv::imread(img_path);
        if (!img.empty()) {
            images.push_back(img);
        } else {
            std::cerr << "无法读取图片: " << img_path << std::endl;
        }
    }

    std::vector<yolo::Image> yoloimages(images.size());
    std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);
    auto batched_result = yoloModel_->forwards(yoloimages);
    for (int ib = 0; ib < (int)batched_result.size(); ++ib) {
        auto &objs = batched_result[ib];
        auto &image = images[ib];
        for (auto &obj : objs) {
            uint8_t b, g, r;
            tie(b, g, r) = yolo::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                          cv::Scalar(b, g, r), 5);

            auto name = labels_[obj.class_label];
            if (std::find(names.begin(), names.end(), name) == names.end()) {
                continue;
            }
            auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
            int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                          cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                        16);
        }
//        printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
//        cv::imwrite(cv::format("Result%d.jpg", ib), image);
    }
    return images;
}