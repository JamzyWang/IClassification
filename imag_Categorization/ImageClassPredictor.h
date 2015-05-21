#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <omp.h>

class ImageClassPredictor
{
private:
    cv::Ptr<cv::FeatureDetector > feature_detector;
    cv::Ptr<cv::BOWImgDescriptorExtractor > bowide;
    cv::Ptr<cv::DescriptorMatcher > descriptor_matcher;
    cv::Ptr<cv::DescriptorExtractor > descriptor_extractor;
    std::map<std::string,CvSVM> classes_classifiers;
    cv::Mat vocabulary;

private:
    bool LoadVocabulary(std::string file_name);
    bool LoadSVMClassifiers(std::string file_prefix);

public:
    ImageClassPredictor(std::string vocabulary_file_name, std::string classifier_file_prefix);
    ~ImageClassPredictor(void);

public:
    std::string PredictClass(cv::Mat& input_image);
};

