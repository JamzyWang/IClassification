#include "SVMPredictor.h"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <cstdlib>

using namespace cv;
using namespace std;

// resize to 256x256
void ProcessImage(const cv::Mat& image, cv::Mat& unified_image)
{
    int s = cv::min(image.rows, image.cols);
    float scale = 256.0 / s;
    scale = scale < 1.0 ? scale : 1.0;
    Size size(image.cols  * scale, image.rows * scale);
    cv::resize(image, unified_image, size);
    cv::GaussianBlur(unified_image, unified_image, Size(11, 11), 5.0);
}

SVMPredictor::SVMPredictor(std::string vovabulary_file_name, std::string svm_classifier_file_name):
    feature_detector(new SurfFeatureDetector()), 
    descriptor_matcher(new BruteForceMatcher<L2<float>>),
    //descriptor_matcher(new FlannBasedMatcher()),
    descriptor_extractor(new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new SurfDescriptorExtractor())))
{
    bool ret = LoadVocabulary(vovabulary_file_name);
    assert(ret);
    ret = LoadSVMClassifier(svm_classifier_file_name);
    assert(ret);
    bowide = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(descriptor_extractor, descriptor_matcher));
    bowide->setVocabulary(vocabulary);
}


SVMPredictor::~SVMPredictor(void)
{
}

bool SVMPredictor::LoadVocabulary(std::string file_name)
{
    try
    {
        FileStorage fs(file_name, FileStorage::READ);
        fs["vocabulary"] >> vocabulary;
        fs.release();
    }
    catch(...)
    {
        cerr << "LoadVocabulary error" << endl;
        return false;
    }
    if (vocabulary.rows == 0)
        return false;
    return true;
}

bool SVMPredictor::LoadSVMClassifier(std::string file_name)
{
    try
    {
        svm.load(file_name.c_str());
    }
    catch (...)
    {
        cerr << "LoadSVMClassifiers error" << endl;
        return false;
    }
    return true;
}

std::string SVMPredictor::PredictClass(cv::Mat& input_image)
{
    Mat unified_image;
    ProcessImage(input_image, unified_image);

    vector<KeyPoint> keypoints;
    Mat response_hist;
    feature_detector->detect(unified_image, keypoints);
    bowide->compute(unified_image, keypoints, response_hist);
    int predict_result = (int)svm.predict(response_hist);

    char class_buf[8];
    itoa((int)predict_result, class_buf, 10);
    return class_buf;
}