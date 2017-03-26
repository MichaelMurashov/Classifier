cmake_minimum_required(VERSION 3.6)
project(Classifier)

set(CMAKE_CXX_STANDARD 11)

set(LIBRARY "_Classifier")
add_subdirectory(src)

set(SOURCE_FILES main.cpp)
add_executable(Classifier ${SOURCE_FILES})

target_link_libraries(Classifier ${LIBRARY})
