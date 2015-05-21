#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <iomanip>
#include <sstream>
#include "ImageClassPredictor.h"
#include "SVMPredictor.h"

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

#define _LOG_RESULT_ 1

std::string image_folder_path("C:\\Users\\Hongze Zhao\\Downloads\\MLKD-Final-Project-Release\\ic-data\\extra\\");

bool ReadImageNamesAndLabels(std::vector<std::string>& image_file_names, std::vector<std::string>& image_labels, std::string list_file_name)
{
    using namespace std;
    ifstream label_list_file(image_folder_path + list_file_name);
    if (!label_list_file)
        return false;
    string fname;
    string label;
    while (!label_list_file.eof())
    {
        label_list_file >> fname >> label;
        if (fname.length() == 0 || label.length() == 0)
            continue;
        image_file_names.push_back(fname);
        image_labels.push_back(label);
    }
    label_list_file.close();
    return true;
}

bool ReadImageNames(std::vector<std::string>& image_file_names, std::string list_file_name)
{
    using namespace std;
    ifstream label_list_file(image_folder_path + list_file_name);
    if (!label_list_file)
        return false;
    string fname;
    string line;
    while (!label_list_file.eof())
    {
        getline(label_list_file, line);
        stringstream ss(line);
        ss >> fname;
        if (fname.length() == 0)
            continue;
        image_file_names.push_back(fname);
    }
    label_list_file.close();
    return true;
}

int main(int argc, char* argv[])
{
    using namespace cv;
    using namespace std;

    if (argc != 5)
    {
        std::cout << "Usage: " << argv[0] << " <vocabulary_file_name> <SVM_Classifier_file_prefix> <image folder> <image_name_list>" << endl;
        std::cout << " eg. " << argv[0] << "vocabulary_color_surf_2000.yml SVM_classifier_color_blur_flann_ \"C:\\images\\\" test.list" << endl;
        return -1;
    }

    image_folder_path = argv[3];
    vector<string> image_file_names;
    
    #if _LOG_RESULT_
    ReadImageNames(image_file_names, argv[4]);
    #else
    vector<string> image_labels;
    ReadImageNamesAndLabels(image_file_names, image_labels, argv[4]);
    #endif

    ImageClassPredictor predictor(argv[1], argv[2]);
    //SVMPredictor predictor("vocabulary_color_blur_500.yml", "SVM_classifier_color_blur_brute_multi.yml");

    std::cout << "predicting images ...         " << std::endl;
    map<std::string, int> correct_counts;
    map<std::string, int> total_counts;
    map<std::string, map<std::string, int>> confusion_matrix;
    char buff_i[4];
    char buff_j[4];
    for (int i = 1; i <= 10; i++)
    {
        itoa(i, buff_i, 10);
        for (int j = 1; j <= 10; j++)
        {
            itoa(j, buff_j, 10);
            confusion_matrix[buff_i][buff_j] = 0;
        }
    }

    #if _LOG_RESULT_
    ofstream output("predict_results.txt", ios::out);
    #endif

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < image_file_names.size(); i++)
    {
        #if !(_LOG_RESULT_)
        string& class_label = image_labels[i];
        #endif

        Mat image = imread(image_folder_path + image_file_names[i] + ".jpg");

        if (image.rows == 0 || image.cols == 0)
        {
            cout << endl << image_file_names[i] << " read error!" << endl;
            continue;
        }

        string predict_label = predictor.PredictClass(image);
        
        
        #pragma omp critical
        {
            std::cout << "\b\b\b\b\b\b\b\b\b";
            std::cout << setfill(' ') << setw(4) << i + 1 << "/" << setw(4) << image_file_names.size();

            
            #if _LOG_RESULT_
            output << image_file_names[i] << "\t" << predict_label << endl;
            #else
            bool predict_correct = predict_label == class_label;

            correct_counts[class_label] += (predict_correct ? 1 : 0);
            total_counts[class_label]++;

            confusion_matrix[class_label][predict_label]++;

            std::cout << " correct? " << (predict_correct ? "yes" : "no") << " (" << predict_label << " - " << class_label << ")" << endl;
            #endif
        }
    }
    std::cout << endl;
    #if _LOG_RESULT_
    output.close();
    #else
    // print confusion matrix
    for (map<std::string, map<std::string, int>>::iterator real_ite = confusion_matrix.begin(); real_ite != confusion_matrix.end(); ++real_ite)
    {
        float class_total = (float)total_counts[real_ite->first];
        for (map<std::string, int>::iterator predict_ite = real_ite->second.begin(); predict_ite != real_ite->second.end(); ++ predict_ite)
        {
            std::cout << setw(4) << predict_ite->second / class_total * 100.0 << " ";
        }
        std::cout << endl;
    }

    // print evaluate result
    int total_correct = 0;
    for (map<std::string, int>::iterator ite = correct_counts.begin(); ite != correct_counts.end(); ++ite)
    {
        int class_total = total_counts[ite->first];
        float percent = (float)ite->second / (float)class_total * 100.0;
        std::cout << "Class " << ite->first << ": " << ite->second << " / " << class_total << " (" << percent << " %)" << endl;
        total_correct += ite->second;
    }
    float total_percent = (float)total_correct / (float)image_file_names.size() * 100.0;
    std::cout << endl << "Total correct:" << total_correct << " / " << image_file_names.size() << " (" << total_percent << " %)" << endl;
    #endif

    ::system("pause");
    return 0;
}