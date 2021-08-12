cmake_minimum_required(VERSION 3.10)
project(demo)
set(CMAKE_CXX_FLAGS "-std=c++11 -O0")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-maybe-uninitialized -O3 -march=native")
find_package(OpenCV 3.4.8 QUIET)
find_package(CUDA REQUIRED)
#这里用的是CUDA10.0+CUDNN7.6.5
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(/home/micang/anaconda3/envs/tf1/lib)
link_directories(/usr/local/lib)

include_directories(
        /home/micang/tensorflow-r1.15/tensorflow/contrib/makefile/downloads/absl
        /home/micang/tensorflow-r1.15/tensorflow_include
) #tensorflow编译生成的库文件路径
add_library(${PROJECT_NAME} SHARED
        tf_model.cpp)

add_executable(main main.cpp)

# 以下是将可执行文件与一些.so文件建立动态链接关系，
# 用到的有libtensorflow_cc.so，libtensorflow_framework.so,以及opencv相关的so
target_link_libraries(main ${PROJECT_NAME}
        libtensorflow_cc.so
        libtensorflow_framework.so
        ${OpenCV_LIBS}
)
#需要在环境变量中设置：export CUDA_VISIBLE_DEVICES='0'