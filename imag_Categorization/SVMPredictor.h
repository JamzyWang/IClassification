#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <omp.h>

class SVMPredictor
{
private:
    cv::Ptr<cv::FeatureDetector > feature_detector;
    cv::Ptr<cv::BOWImgDescriptorExtractor > bowide;
    cv::Ptr<cv::DescriptorMatcher > descriptor_matcher;
    cv::Ptr<cv::DescriptorExtractor > descriptor_extractor;
    CvSVM svm;
    cv::Mat vocabulary;
private:
    bool LoadVocabulary(std::string file_name);
    bool LoadSVMClassifier(std::string file_name);
public:
    SVMPredictor(std::string vovabulary_file_name, std::string svm_classifier_file_name);
    ~SVMPredictor(void);
    std::string PredictClass(cv::Mat& input_image);
};

