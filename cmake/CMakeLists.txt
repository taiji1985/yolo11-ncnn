cmake_minimum_required(VERSION 3.14)

# 项目名称和版本
project(ncnn_project VERSION 1.0)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置编译选项（如启用优化）
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug Build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")
else()
    message("Release Build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# 添加子目录（如果有更多模块可以扩展）
add_subdirectory(src)
