#include <iostream>

#include "src/bow.cpp"
#include "src/auxiliary.cpp"

//

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
    Ptr<Feature2D> keyPointsDetector = SURF::create();

    Ptr<DescriptorMatcher> dMatcher = DescriptorMatcher::create("BruteForce");

    Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor(dMatcher);

    vector<string> trainFilesList, testFilesList;
    Mat trainAnswers, testAnswers;

    // получаем списки файлов и правильные ответы для выборок
    getAnswers(argv, trainAnswers, testAnswers, trainFilesList, testFilesList);

    cout << "train count: " << trainFilesList.size() << endl
         << "test count: " << testFilesList.size() << endl;

    // устанавливаем словарь
    Mat vocabulary = trainVocabulary(trainFilesList, keyPointsDetector);
    bowExtractor->setVocabulary(vocabulary);

    // формируем обучающую выборку
    Mat trainData, trainResponses;
    extractTrainData(trainFilesList, trainAnswers, trainData, trainResponses, keyPointsDetector, bowExtractor);

    // обучаем классификатор
    Ptr<RTrees> rTrees = trainClassifier(trainData, trainResponses);

    // получаем набор предсказанный значений
    Mat predictions = predictOnTestData(testFilesList, keyPointsDetector, bowExtractor, rTrees);

    // считаем ошибку
    float error = calcClassificationError(testAnswers, predictions);
    cout << 100 - error * 100 << "% correct answers" << endl;

    return 0;
}
