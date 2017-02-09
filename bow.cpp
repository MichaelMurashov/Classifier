#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
//#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int vocSize = 10;

// обучение словоря
Mat trainVocabulary(const vector<string>& filesList, const Ptr<Feature2D>& keyPointsDetector) {
    Mat img, descriptors;
    vector<KeyPoint> keyPoints;
    BOWKMeansTrainer tr(vocSize);

    for (int i = 0; i < filesList.size(); i++) {
        img = imread(filesList[i], IMREAD_GRAYSCALE);

        keyPointsDetector->detect(img, keyPoints);
        keyPointsDetector->compute(img, keyPoints, descriptors);

        tr.add(descriptors);
    }

    return tr.cluster();
}

// возвращает признаковое описание изображения
Mat ExtractFeaturesFromImage(const Ptr<Feature2D>& keyPointsDetector, const Ptr<BOWImgDescriptorExtractor>& bowExtractor,
                             const string& fileName) {
    Mat img = imread(fileName, IMREAD_GRAYSCALE);

    vector<KeyPoint> keyPoints;
    keyPointsDetector->detect(img, keyPoints);

    Mat imgDescriptor;
    bowExtractor->compute(img, keyPoints, imgDescriptor);

    return imgDescriptor;
}

// формирование обучающей выборки
void ExtractTrainData(const vector<string>& filesList, const Mat& responses, Mat& trainData, Mat& trainResponses,
                      const Ptr<Feature2D>& keyPointsDetector, const Ptr<BOWImgDescriptorExtractor>& bowExtractor) {
    trainData.create(filesList.size(), bowExtractor->descriptorSize(), CV_32F);
    trainResponses.create(filesList.size(), 1, CV_32S);

    for (int i = 0; i < filesList.size(); i++) {
        trainData.push_back(ExtractFeaturesFromImage(keyPointsDetector, bowExtractor, filesList[i]));
        trainResponses.push_back(responses.at<int>(i));
    }
}


