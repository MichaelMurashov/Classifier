#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace cv::xfeatures2d;

const int vocSize = 25;
const int numOfTrees = 200;

// обучение словоря
Mat trainVocabulary(const vector<string>& filesList, const Ptr<Feature2D>& keyPointsDetector) {
    Mat img, descriptors;
    vector<KeyPoint> keyPoints;
    BOWKMeansTrainer tr(vocSize);

    for (int i = 0; i < filesList.size(); i++) {
       // cout << i << endl;
        img = imread(filesList[i], IMREAD_GRAYSCALE);

        keyPointsDetector->detect(img, keyPoints);
        keyPointsDetector->compute(img, keyPoints, descriptors);

        tr.add(descriptors);
    }

    Mat qwe = tr.cluster();
    return qwe;
}

// возвращает признаковое описание изображения
//Mat extractFeaturesFromImage(Ptr<Feature2D> keyPointsDetector, Ptr<BOWImgDescriptorExtractor> bowExtractor,
//                             const string& fileName) {
//    Mat img = imread(fileName, IMREAD_GRAYSCALE);
//
//    vector<KeyPoint> keyPoints;
//    keyPointsDetector->detect(img, keyPoints);
//
//    Mat imgDescriptor;
//    bowExtractor->compute(img, keyPoints, imgDescriptor);
//
//    return imgDescriptor;
//}

// формирование обучающей выборки
void extractTrainData(const vector<string>& filesList, const Mat& responses, Mat& trainData, Mat& trainResponses,
                      const Ptr<Feature2D>& keyPointsDetector, const Ptr<BOWImgDescriptorExtractor>& bowExtractor) {
    Mat img, imgDescriptor;
    vector<KeyPoint> keyPoints;

    trainData.create(filesList.size(), bowExtractor->descriptorSize(), CV_32F);
    trainResponses.create(filesList.size(), 1, CV_32S);

    for (int i = 0; i < filesList.size(); i++) {
        img = imread(filesList[i], IMREAD_GRAYSCALE);

        keyPointsDetector->detect(img, keyPoints);
        bowExtractor->compute(img, keyPoints, imgDescriptor);

        trainData.push_back(extractFeaturesFromImage(keyPointsDetector, bowExtractor, filesList[i]));
        trainResponses.push_back(responses.at<int>(i));
    }
}

// обучение классификатора «случайный лес»
Ptr<RTrees> trainClassifier(const Mat& trainData, const Mat& trainResponses) {
    Ptr<RTrees> rTrees;

    rTrees = ml::RTrees::create();
    rTrees->setTermCriteria(TermCriteria(TermCriteria::COUNT, numOfTrees, 0));

    Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainResponses);

    rTrees->train(tData);

    return rTrees;
}

// возвращает набор предсказанных значений для тестовой выборки
Mat predictOnTestData(const vector<string>& filesList, const Ptr<Feature2D> keyPointsDetector,
                      const Ptr<BOWImgDescriptorExtractor> bowExtractor, const Ptr<RTrees> classifier) {
    Mat answers(filesList.size(), 1, CV_32S);

    for (int i = 0; i < filesList.size(); i++) {
        Mat description = extractFeaturesFromImage(keyPointsDetector, bowExtractor, filesList[i]);

        answers.push_back(classifier->predict(description));
    }

    return answers;
}


