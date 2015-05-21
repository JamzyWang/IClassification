#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_core245d")
#pragma comment(lib, "opencv_highgui245d")
#pragma comment(lib, "opencv_features2d245d")
#pragma comment(lib, "opencv_ml245d")
#pragma comment(lib, "opencv_nonfree245d")
#pragma comment(lib, "opencv_imgproc245d")
#else
#pragma comment(lib, "opencv_core245")
#pragma comment(lib, "opencv_highgui245")
#pragma comment(lib, "opencv_features2d245")
#pragma comment(lib, "opencv_ml245")
#pragma comment(lib, "opencv_nonfree245")
#pragma comment(lib, "opencv_imgproc245")
#endif

std::string image_folder_path("C:\\Users\\Hongze Zhao\\Downloads\\MLKD-Final-Project-Release\\ic-data\\train\\");

bool ReadImageNames(std::vector<std::string>& image_file_names, std::string folder, std::string list_file_name)
{
    using namespace std;
    ifstream label_file(folder + list_file_name);
    if (!label_file)
        return false;
    string fname;
    while (!label_file.eof())
    {
        label_file >> fname;
        if (fname.length() == 0)
            continue;
        image_file_names.push_back(folder + fname + ".jpg");
        label_file >> fname;
    }
    label_file.close();
    return true;
}

// resize to 256x256
void UnifyImageSize(cv::Mat& image)
{
    using namespace cv;
    Mat unified_image;
    int s = cv::min(image.rows, image.cols);
    float scale = 128.0 / s;
    Size size(image.cols  * scale, image.rows * scale);
    cv::resize(image, unified_image, size);
    cv::medianBlur(unified_image, unified_image, 3);
    image = unified_image;
}

int main(int argc, char* argv[])
{
    using namespace cv;
    using namespace std;

    if (argc != 3)
    {
        cout << "USAGE: " << argv[0] << " <folder> <image list file>" << endl;
        return -1;
    }
    image_folder_path = argv[1];
    string image_list_file = argv[2];
    if (image_folder_path.back() != '\\' || image_folder_path.back() != '/')
        image_folder_path += "\\";

    int minHessian = 400;
    //Ptr<FeatureDetector> feature_detector(new SurfFeatureDetector(minHessian));
    Ptr<FeatureDetector> feature_detector = Ptr<cv::FeatureDetector>(new PyramidAdaptedFeatureDetector(Ptr<cv::FeatureDetector>(new SurfFeatureDetector(minHessian))));
    //Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create("GridFAST");
    Ptr<DescriptorExtractor> descriptor_extractor(new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new SurfDescriptorExtractor(minHessian))));
    //Ptr<DescriptorExtractor> descriptor_extractor(new SurfDescriptorExtractor());
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat training_descriptors;
    int i;

    cout << "------ build vocabulary -----" << endl;

    cout << "reading " << image_list_file << " ..." << endl;
    vector<string> image_file_names;
    if(!ReadImageNames(image_file_names, image_folder_path, image_list_file))
    {
        cerr << "cannot read image names" << endl;
        return -1;
    }

    Mat image;
    cout << "extract descriptors ...         ";
    #pragma omp parallel for schedule(dynamic) private(image, keypoints, descriptors)
    for (i = 0; i < image_file_names.size(); i++)
    {
        image = imread(image_file_names[i]);//, CV_LOAD_IMAGE_GRAYSCALE);
        UnifyImageSize(image);
        feature_detector->detect(image, keypoints);
        descriptor_extractor->compute(image, keypoints, descriptors);
        #pragma omp critical
        {
            training_descriptors.push_back(descriptors);
            cout << "\b\b\b\b\b\b\b\b\b";
            cout << setfill(' ') << setw(4) << i << "/" << setw(4) << image_file_names.size();
        }
    }
    cout << endl;

    cout << "Total Descriptors: " << training_descriptors.rows << endl;
    cout << "Saving training_descriptors.yml" << endl;

    FileStorage fs_descriptors("training_descriptors.yml", FileStorage::WRITE);
    fs_descriptors << "training_descriptors" << training_descriptors;
    fs_descriptors.release();

    BOWKMeansTrainer bowtrainer(2000); // 1000 clusters
    bowtrainer.add(training_descriptors);
    cout << "clustering BOW features ..." << endl;
    Mat vocabulary = bowtrainer.cluster();

    cout << "Saving vocabulary_color_crop_2000.yml" << endl;
    FileStorage fs_vocabulary("vocabulary_color_surf_2000.yml", FileStorage::WRITE);
    fs_vocabulary << "vocabulary" << vocabulary;
    fs_vocabulary.release();

    return 0;
}