# 指定源文件
set(SOURCES
    yolov8.cpp
)


# set(OpenCV_DIR "F:/study/opencv-windows-4.10.0/opencv/build") # 替换为 OpenCV 的实际路径
set(OpenCV_DIR "D:/libs/opencv345/x64/vc15/lib")

# 指定头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 创建可执行文件
add_executable(ncnn_project ${SOURCES})

set(ncnn_DIR "F:/study/ncnn-20240820-windows-vs2022-shared/x64/lib/cmake/ncnn") # 替换为 ncnn 的实际路径

# 链接 ncnn 库和 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(ncnn REQUIRED)
include_directories(${ncnn_INCLUDE_DIRS})

target_link_libraries(ncnn_project ${OpenCV_LIBS} ncnn)
