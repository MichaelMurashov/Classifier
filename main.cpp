#include <iostream>
#include <dirent.h>

#include "bow.cpp"

void getFilesInDir(const string& dirPath, vector<string>& filesList, int& count) {
    DIR* dir = opendir(dirPath.c_str());  // открываем директорию
    if (dir == NULL) {
        cout << "Can't open the directory: " << dirPath;
        exit(0);
    }

    // читаем файлы
    dirent* dirent;
    while ((dirent = readdir(dir)) != NULL) {
        char* name = dirent->d_name;
        if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0 || strcmp(name, ".DS_Store") == 0)
            continue;

        count++;
        filesList.push_back(dirPath + "/" + (const char*)name);  // заносим в список
    }

    closedir(dir);  // закрываем директорию
}

int main(int argc, char* argv[]) {
    const int numOfCategory = 4;
    vector<string> trainFilesList, testFilesList;

    // считываем тренировочную выборку
    int trainCategory[numOfCategory];
    for (int i = 1; i <= numOfCategory; i++) {
        int count = 0;
        getFilesInDir(argv[i], trainFilesList, count);
        trainCategory[i] = count;
    }

    // формируем правильные ответы для тренировочной выборки
    Mat trainAnswers((int)trainFilesList.size(), 1, CV_32S);
    for (int i = 0; i < numOfCategory; i++) {
        for (int j = 0; j < trainCategory[i]; i++) {
            trainAnswers.push_back(i);
        }
    }

    // считываем тестовую выборку
    int testCategory[numOfCategory];
    for (int i = 1; i <= numOfCategory; i++) {
        int count = 0;
        getFilesInDir(argv[i], testFilesList, count);
        testCategory[i] = count;
    }

    // формируем правильные ответы для тестовой выборки
    Mat testAnswers((int)testFilesList.size(), 1, CV_32S);
    for (int i = 0; i < numOfCategory; i++) {
        for (int j = 0; j < testCategory[i]; i++) {
            testAnswers.push_back(i);
        }
    }

    Ptr<Feature2D> keyPointsDetector = SIFT::create();

    Ptr<DescriptorMatcher> dMatcher = DescriptorMatcher::create("BruteForce");

    Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor(dMatcher);

    Mat vocabulary = trainVocabulary(trainFilesList, keyPointsDetector);
    bowExtractor->setVocabulary(vocabulary);

    // формируем обучающую выборку
    Mat trainData, trainResponses;
    ExtractTrainData(trainFilesList, trainAnswers, trainData, trainResponses, keyPointsDetector, bowExtractor);



    return 0;
}
