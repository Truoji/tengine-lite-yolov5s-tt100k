cmake_minimum_required(VERSION 3.10) 
project(yolov5s-tt100k)  

set(CMAKE_CXX_STANDARD 11)  
find_package(OpenCV REQUIRED) 

SET(CMAKE_BUILD_TYPE "Debug")

include_directories(include/)  

add_library(tengine SHARED IMPORTED) 
set_target_properties(tengine PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/third_party/libtengine-lite.so)  

add_executable(yolov5s-tt100k ${CMAKE_SOURCE_DIR}/src/yolov5s-tt100k-cpu.cpp) 
target_link_libraries(yolov5s-tt100k tengine ${OpenCV_LIBS})

# add_executable(yolov5s-tt100k_int8 ${CMAKE_SOURCE_DIR}/src/yolov5s-tt100k-int8.cpp) 
# target_link_libraries(yolov5s-tt100k_int8 tengine ${OpenCV_LIBS})

# add_executable(yolov5s-tt100k_uint8 ${CMAKE_SOURCE_DIR}/src/yolov5s-tt100k-uint8.cpp) 
# target_link_libraries(yolov5s-tt100k_uint8 tengine ${OpenCV_LIBS})
