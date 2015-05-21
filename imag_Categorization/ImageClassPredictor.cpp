#include "ImageClassPredictor.h"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <cstdlib>

using namespace cv;
using namespace std;

ImageClassPredictor::ImageClassPredictor(std::string vocabulary_file_name, std::string classifier_file_prefix): 
    feature_detector(new SurfFeatureDetector(400)), 
    //descriptor_matcher(new BruteForceMatcher<L2<float>>),
    descriptor_matcher(new FlannBasedMatcher()),
    descriptor_extractor(new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new SurfDescriptorExtractor(400))))
{
    bool ret = LoadVocabulary(vocabulary_file_name);
    assert(ret);
    ret = LoadSVMClassifiers(classifier_file_prefix);
    assert(ret);
    //feature_detector = Ptr<cv::FeatureDetector>(new PyramidAdaptedFeatureDetector(Ptr<cv::FeatureDetector>(new SurfFeatureDetector(400))));
    bowide = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(descriptor_extractor, descriptor_matcher));
    bowide->setVocabulary(vocabulary);
    descriptor_matcher->train();
}

ImageClassPredictor::~ImageClassPredictor(void)
{
}

bool ImageClassPredictor::LoadVocabulary(std::string file_name)
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

bool ImageClassPredictor::LoadSVMClassifiers(std::string file_prefix)
{
    const int classes_count = 10; // totally 10 classes of images
    classes_classifiers.clear();
    try
    {
        for (int i = 0; i < classes_count; i++)
        {
            char class_name[4];
            itoa(i + 1, class_name, 10);
            string file_name = file_prefix + class_name + ".yml";
            classes_classifiers.insert(pair<string, CvSVM>(class_name, CvSVM()));
            classes_classifiers[class_name].load(file_name.c_str());
        }
    }
    catch (...)
    {
        cerr << "LoadSVMClassifiers error" << endl;
        return false;
    }
    return true;
}

// resize to 256x256
void UnifyImageSize(const cv::Mat& image, cv::Mat& unified_image)
{
    int s = cv::min(image.rows, image.cols);
    //float scale = 256.0 / s;
    //Size size(image.cols  * scale, image.rows * scale);
    Size size(128, 128);
    cv::resize(image, unified_image, size);
    //cv::GaussianBlur(unified_image, unified_image, Size(3, 3), 3.0);
    //cv::filter2D(unified_image, unified_image, unified_image.depth(), CV_MEDIAN);
    cv::medianBlur(unified_image, unified_image, 3);
}

std::string ImageClassPredictor::PredictClass(Mat& input_image)
{
    Mat unified_image;
    UnifyImageSize(input_image, unified_image);

    // sliding window approach
    const int window_size = 128; // unified_image.rows < 128 || unified_image.cols < 128 ? cv::min(unified_image.cols, unified_image.rows) : 128;
    vector<Point> check_points;
    for (int i = 0; i < unified_image.cols; i += window_size / 4)
        for (int j = 0; j < unified_image.rows; j += window_size / 4)
            check_points.push_back(Point(i, j));

    map<string,pair<int,float> > found_classes;

    #pragma omp parallel for
    for (int i = 0; i < check_points.size(); i++)
    {
        Point& p = check_points[i];
        // crop window image
        Mat image;
        unified_image(Rect(p.x - window_size/2, p.y - window_size/2, window_size, window_size) & Rect(0, 0, unified_image.cols, unified_image.rows)).copyTo(image);

        if (image.rows == 0 || image.cols == 0)
            continue;
        // detect keypoints
        vector<KeyPoint> keypoints;
        Mat response_hist;
        feature_detector->detect(image, keypoints);
        bowide->compute(image, keypoints, response_hist);
        if (response_hist.cols == 0 || response_hist.rows == 0)
            continue;
        
        // predict window image
        try
        {
            float min_distance = FLT_MAX;
            string min_class = "!";
            for (map<string, CvSVM>::iterator ite = classes_classifiers.begin(); ite != classes_classifiers.end(); ++ite)
            {
                // signed distance to the margin (support vector)
                float predict_distance = ite->second.predict(response_hist, true);
                if (predict_distance > 1.0)
                    continue;
                if (predict_distance < min_distance)
                {
                    min_distance = predict_distance;
                    min_class = ite->first;
                }
            }
            if (min_class == "!")
                continue;
            #pragma omp critical
            {
                found_classes[min_class].first++;
                found_classes[min_class].second += min_distance;
            }
        }
        catch (cv::Exception)
        {
            continue;
        }
    }

    // get the best matched class
    float max_class_score = -FLT_MAX;
    string max_class;
    //cout << " -->> ";
    for (map<string,pair<int,float> >::iterator ite = found_classes.begin(); ite != found_classes.end(); ++ite)
    {
        float score = abs(ite->second.first * ite->second.second);
        //cout << ite->first << ", " << score << " | ";
        if (score > 1e10)
            continue;   // impossible score
        if (score > max_class_score)
        {
            max_class_score = score;
            max_class = ite->first;
        }
    }
    //cout << endl;
    //cout << "max_score = " << max_class_score << endl;
    return max_class;
}