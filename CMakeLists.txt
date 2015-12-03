cmake_minimum_required(VERSION 3.3)
project(texture_classification)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES  src/LBP.hpp src/LBP.cpp src/histogram.hpp src/histogram.cpp  src/MyData.h src/MyData.cpp  main.cpp )
find_package(OpenCV REQUIRED)
add_executable(texture_classification ${SOURCE_FILES} main.cpp)
target_link_libraries(texture_classification ${OpenCV_LIBS})