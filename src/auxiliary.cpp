#include <vector>
#include <dirent.h>

#include <opencv2/opencv.hpp>

using namespace std;
using cv::Mat;

const int NUM_OF_CATEGORY = 4;

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

float calcClassificationError(Mat& responses, Mat& predictions) {
    float size = (float)responses.rows;
    float error = 0.0;

    for (int i = 0; i < size; i++) {
        float c1 = (float)(responses.at<int>(i));
        float c2 = predictions.at<float>(i);

        if (c1 != c2)
            error++;
    }

    return error / size;
}

void getAnswers(char* argv[], Mat& trainAnswers, Mat& testAnswers,
                vector<string>& trainFilesList, vector<string>& testFilesList) {
    int trainCategories[NUM_OF_CATEGORY], testCategories[NUM_OF_CATEGORY];

    for (int i = 0; i < NUM_OF_CATEGORY; i++) {
        int trainCount = 0;
        int testCount = 0;

        getFilesInDir(argv[i + 1], trainFilesList, trainCount);
        getFilesInDir(argv[i + 1 + NUM_OF_CATEGORY], testFilesList, testCount);

        trainCategories[i] = trainCount;
        testCategories[i] = testCount;
    }

    trainAnswers.create(0, 1, CV_32S);
    for (int i = 0; i < NUM_OF_CATEGORY; i++)
        for (int j = 0; j < trainCategories[i]; j++) {
            trainAnswers.push_back(i);
        }

    testAnswers.create(0, 1, CV_32S);
    for (int i = 0; i < NUM_OF_CATEGORY; i++)
        for (int j = 0; j < testCategories[i]; j++) {
            testAnswers.push_back(i);
        }
}
