#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace cv::xfeatures2d;

const int VOC_SIZE = 25;
const int NUM_OF_TREES = 500;

// обучение словоря
Mat trainVocabulary(const vector<string>& filesList, const Ptr<Feature2D>& keyPointsDetector) {
    Mat img, descriptors;
    vector<KeyPoint> keyPoints;
    BOWKMeansTrainer tr(VOC_SIZE);

    cout << endl << "Train vocabulary" << endl;
    for (int i = 0; i < filesList.size(); i++) {
        img = imread(filesList[i], IMREAD_GRAYSCALE);

        keyPointsDetector->detect(img, keyPoints);
        keyPointsDetector->compute(img, keyPoints, descriptors);

        tr.add(descriptors);
    }

    return tr.cluster();
}

// возвращает признаковое описание изображения
Mat extractFeaturesFromImage(Ptr<Feature2D> keyPointsDetector, Ptr<BOWImgDescriptorExtractor> bowExtractor,
                             const string& fileName) {
    Mat imgDescriptor, keyPointsDescriptors;
    Mat img = imread(fileName, IMREAD_GRAYSCALE);
    vector<KeyPoint> keyPoints;

    keyPointsDetector->detect(img, keyPoints);

    keyPointsDetector->compute(img, keyPoints, keyPointsDescriptors);
    bowExtractor->compute(keyPointsDescriptors, imgDescriptor);

    return imgDescriptor;
}

// формирование обучающей выборки
void extractTrainData(const vector<string>& filesList, const Mat& responses, Mat& trainData, Mat& trainResponses,
                      const Ptr<Feature2D>& keyPointsDetector, const Ptr<BOWImgDescriptorExtractor>& bowExtractor) {
    trainData.create(0, bowExtractor->descriptorSize(), CV_32F);
    trainResponses.create(0, 1, CV_32S);

    cout << "Formation of training sample" << endl;
    for (int i = 0; i < filesList.size(); i++) {
        trainData.push_back(extractFeaturesFromImage(keyPointsDetector, bowExtractor, filesList[i]));
        trainResponses.push_back(responses.at<int>(i));
    }
}

// обучение классификатора «случайный лес»
Ptr<RTrees> trainClassifier(const Mat& trainData, const Mat& trainResponses) {
    Ptr<RTrees> rTrees;

    rTrees = RTrees::create();
    rTrees->setTermCriteria(TermCriteria(TermCriteria::COUNT, NUM_OF_TREES, 0));

    Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainResponses);

    rTrees->train(tData);

    return rTrees;
}

// возвращает набор предсказанных значений для тестовой выборки
Mat predictOnTestData(const vector<string>& filesList, const Ptr<Feature2D> keyPointsDetector,
                      const Ptr<BOWImgDescriptorExtractor> bowExtractor, const Ptr<RTrees> classifier) {
    Mat answers(0, 1, CV_32F);

    cout << "Prediction class";
    for (int i = 0; i < filesList.size(); i++) {
        Mat description = extractFeaturesFromImage(keyPointsDetector, bowExtractor, filesList[i]);

        float answer = classifier->predict(description);
        answers.push_back(answer);
    }

    cout << endl << endl;
    return answers;
}
