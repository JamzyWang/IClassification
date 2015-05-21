#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_core245d")
#pragma comment(lib, "opencv_highgui245d")
#pragma comment(lib, "opencv_features2d245d")
#pragma comment(lib, "opencv_ml245d")
#pragma comment(lib, "opencv_nonfree245d")
#pragma comment(lib, "opencv_legacy245d")
#pragma comment(lib, "opencv_flann245d")
#pragma comment(lib, "opencv_imgproc245d")
#else
#pragma comment(lib, "opencv_core245")
#pragma comment(lib, "opencv_highgui245")
#pragma comment(lib, "opencv_features2d245")
#pragma comment(lib, "opencv_ml245")
#pragma comment(lib, "opencv_nonfree245")
#pragma comment(lib, "opencv_legacy245")
#pragma comment(lib, "opencv_flann245")
#pragma comment(lib, "opencv_imgproc245")
#endif

const std::string image_folder_path("C:\\Users\\Hongze Zhao\\Downloads\\MLKD-Final-Project-Release\\ic-data\\train\\");

bool ReadImageNamesAndLabels(std::vector<std::string>& image_file_names, std::vector<std::string>& image_labels, std::string folder, std::string list_file_name)
{
    using namespace std;
    ifstream label_list_file(folder + list_file_name);
    if (!label_list_file)
        return false;
    string fname;
    string label;
    while (!label_list_file.eof())
    {
        label_list_file >> fname >> label;
        if (fname.length() == 0 || label.length() == 0)
            continue;
        image_file_names.push_back(folder + fname + ".jpg");
        image_labels.push_back(label);
    }
    label_list_file.close();
    return true;
}

// resize to 256x256
void UnifyImageSize(cv::Mat& image)
{
    using namespace cv;
    Mat unified_image;
    int s = cv::min(image.rows, image.cols);
    //float scale = 128.0 / s;
    //Size size(image.cols  * scale, image.rows * scale);
    Size size(128, 128);
    cv::resize(image, unified_image, size);
    //cv::GaussianBlur(unified_image, unified_image, Size(3, 3), 3.0);
    //cv::filter2D(unified_image, unified_image, unified_image.depth(), CV_MEDIAN);
    cv::medianBlur(unified_image, unified_image, 3);
    image = unified_image;
}

void ExtractTrainingSamples(cv::Ptr<cv::FeatureDetector>& detector, cv::BOWImgDescriptorExtractor& bowide, std::map<std::string,cv::Mat>& classes_training_data)
{
    using namespace cv;
    using namespace std;

    vector<string> image_file_names;
    vector<string> image_labels;
    //ReadImageNamesAndLabels(image_file_names, image_labels, image_folder_path, "all.train.label");
    ReadImageNamesAndLabels(image_file_names, image_labels, "C:\\Users\\Hongze Zhao\\Downloads\\MLKD-Final-Project-Release\\ic-data\\extra\\", "extra.label");
    ReadImageNamesAndLabels(image_file_names, image_labels, "C:\\Users\\Hongze Zhao\\Downloads\\MLKD-Final-Project-Release\\ic-data\\check\\", "check.label");

    cout << "extracting training samples ...         ";
    #pragma omp parallel for
    for (int i = 0; i < image_file_names.size(); i++)
    {
        vector<KeyPoint> keypoints;
        Mat response_hist;
        string& class_label = image_labels[i];
        Mat image = imread(image_file_names[i]);//, CV_LOAD_IMAGE_GRAYSCALE);
        UnifyImageSize(image);
        detector->detect(image, keypoints);
        bowide.compute(image, keypoints, response_hist);

        #pragma omp critical
        {
            if (classes_training_data.count(class_label) == 0) // not yet created...
                classes_training_data[class_label].create(0, response_hist.cols, response_hist.type());
            classes_training_data[class_label].push_back(response_hist);

            cout << "\b\b\b\b\b\b\b\b\b";
            cout << setfill(' ') << setw(4) << i << "/" << setw(4) << image_file_names.size();
        }
    }
    cout << endl;

    cout << "saving to file ..." << endl;
    FileStorage fs("training_samples.yml", FileStorage::WRITE);
    for (map<string, Mat>::iterator ite = classes_training_data.begin(); ite != classes_training_data.end(); ++ite)
    {
        cout << "save " << ite->first << endl;
        fs << "class" + ite->first << ite->second;
    }
    fs.release();
}

void TrainSVM(std::map<std::string,cv::Mat>& classes_training_data, std::string& file_postfix, int response_cols, int response_type)
{
    using namespace cv;
    using namespace std;

    vector<string> class_names;
    for (map<string, Mat>::iterator ite = classes_training_data.begin(); ite != classes_training_data.end(); ++ite)
        class_names.push_back(ite->first);

    // one vs. all classifiers
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < class_names.size(); i++)
    {
        string& class_name = class_names[i];
        cout << "training class : " << class_name << " ..." << endl;

        // copy class samples and label
        Mat samples(0, response_cols, response_type);
        Mat labels(0, 1, CV_32FC1); // 0 rows, 1 cols
        samples.push_back(classes_training_data[class_name]);
        Mat class_label = Mat::ones(classes_training_data[class_name].rows, 1, CV_32FC1);
        labels.push_back(class_label);

        // copy rest samples and label
        for (map<string, Mat>::iterator ite = classes_training_data.begin(); ite != classes_training_data.end(); ++ite)
        {
            string not_class_name = ite->first;
            if (not_class_name == class_name)
                continue;
            Mat& not_class_mat = classes_training_data[not_class_name];
            samples.push_back(not_class_mat);
            class_label = Mat::zeros(not_class_mat.rows, 1, CV_32FC1);
            labels.push_back(class_label);
        }

        // train and save
        if (samples.rows == 0)
            continue;
        Mat sample_32f;
        samples.convertTo(sample_32f, CV_32F);
        CvSVMParams svm_param;
        svm_param.svm_type = CvSVM::C_SVC;
        svm_param.kernel_type = CvSVM::RBF;
        //svm_param.nu = 0.5; // in the range 0..1, the larger the value, the smoother the decision boundary
        svm_param.C = 5;
        svm_param.gamma = 0.1;
        //svm_param.degree = 3;
        svm_param.term_crit.epsilon = 1e-8;
        svm_param.term_crit.max_iter = 1e9;
        svm_param.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
        CvSVM svm_classifier;
        svm_classifier.train(sample_32f, labels, Mat(), Mat(), svm_param);
        //svm_classifier.train(sample_32f, labels);
        //svm_classifier.train_auto(sample_32f, labels, Mat(), Mat(), svm_param);

        // save classifier
        string classifier_file_name("SVM_classifier_");
        classifier_file_name += file_postfix + "_" + class_name + ".yml";
        svm_classifier.save(classifier_file_name.c_str());
        cout << classifier_file_name << " saved" << endl;
    }
}

int main(int argc, char* argv[])
{
    using namespace cv;
    using namespace std;

    if (argc < 3) {
        cout << "USAGE: train_bovw <vocabulary_file.yml> <postfix_for_output>"<<endl;
        return -1;
    }

    cout << " ------- Train SVM Classifier -------" << endl;
    // read vocabulary from file
    cout << "reading vocabulary form file ..."<<endl;
    Mat vocabulary;
    FileStorage fs(argv[1], FileStorage::READ);
    fs["vocabulary"] >> vocabulary;
    fs.release();

    if (vocabulary.rows == 0)
    {
        cerr << "Cannot Load Vocabulary File :" << argv[1] << endl;
        return -1;
    }

    // setup BOWImgDescriptorExtractor with vocabulary
    Ptr<FeatureDetector> feature_detector(new SurfFeatureDetector(400));
    //Ptr<FeatureDetector> feature_detector = Ptr<cv::FeatureDetector>(new PyramidAdaptedFeatureDetector(Ptr<cv::FeatureDetector>(new SurfFeatureDetector(400))));
    //Ptr<FeatureDetector> feature_detector = FeatureDetector::create("GridSURF");
    //Ptr<DescriptorExtractor> descriptor_extractor(new SurfDescriptorExtractor());
    Ptr<DescriptorExtractor> descriptor_extractor(new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new SurfDescriptorExtractor(400))));
    Ptr<DescriptorMatcher> descriptor_matcher(new FlannBasedMatcher());
    //Ptr<DescriptorMatcher> descriptor_matcher(new BruteForceMatcher<L2<float>>());
    BOWImgDescriptorExtractor bowide(descriptor_extractor, descriptor_matcher);
    bowide.setVocabulary(vocabulary);

    descriptor_matcher->train(); // call this to load dll, for omp

    // setup training data for classifiers and extract samples from image files
    map<string, Mat> classes_training_data;
    ExtractTrainingSamples(feature_detector, bowide, classes_training_data);

    // show samples information
    cout << "Got " << classes_training_data.size() << " classes." << endl;
    for (map<string, Mat>::iterator ite = classes_training_data.begin(); ite != classes_training_data.end(); ++ite)
        cout << "class " << ite->first << " has " << ite->second.rows << " samples" << endl;

    // train SVM for each classes
    cout << "Training SVMs" << endl;
    string postfix = argv[2];
    TrainSVM(classes_training_data, postfix, bowide.descriptorSize(), bowide.descriptorType());

    system("pause");
    return 0;
}