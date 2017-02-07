#include <iostream>
#include <vector>
#include <dirent.h>

#include <opencv2/features2d/features2d.hpp>

using namespace std;

void getFilesInDir(const string& dirPath, vector<string>& filesList) {
    DIR* dir = opendir(dirPath.c_str());  // открываем общую директорию
    if (dir == NULL) {
        cout << "Can't open the diretcory: " << dirPath;
        exit(0);
    }

    // читаем поддиректории
    dirent* dirent1;
    while ((dirent1 = readdir(dir)) != NULL) {
        // пропускаем скрытые директории
        if (strcmp(dirent1->d_name, ".") == 0 || strcmp(dirent1->d_name, "..") == 0 ||
                strcmp(dirent1->d_name, ".DS_Store") == 0)
            continue;

        string path = dirPath + "/" + (const char*)dirent1->d_name;
        DIR* current = opendir(path.c_str());  // открываем поддиректорию
        if (current == NULL) {
            cout << "Can't open the diretcory: " << dirent1->d_name;
            exit(0);
        }

        // читаем первый файл
        dirent* dirent2;
        int i = 0;
        while ((dirent2 = readdir(current)) != NULL) {
            if (strcmp(dirent2->d_name, ".") == 0 || strcmp(dirent2->d_name, "..") == 0 ||
                    strcmp(dirent2->d_name, ".DS_Store") == 0)
                continue;
            filesList.push_back(path + "/" + (const char*)dirent2->d_name);  // заносим в список
        }

        closedir(current);  // закрываем поддиректорию
    }

    closedir(dir);  // закрываем общую директорию
}

int main(int argc, char** argv) {
    vector<string> testFilesList;
    getFilesInDir(argv[1], testFilesList);



    return 0;
}
