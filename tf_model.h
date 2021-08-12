//
// Created by micang on 2021/7/19.
//
#ifndef DEMO_TF_MODEL_H
#define DEMO_TF_MODEL_H
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/graph/default_device.h"
#include <tensorflow/core/graph/graph_def_builder.h>
#include <iostream>
#include "time.h"
#include "opencv2/opencv.hpp"
class tf_model {
    public:

        void loadModel(const std::string& model_path);

        void CVMat_to_Tensor(cv::Mat& img, tensorflow::Tensor *output_tensor);

        void ge_label(cv::Mat& imSeg);

        void sess(cv::Mat& img);
    public:
        static const int size_w = 576, size_h = 576;
        constexpr static const double iw = 480, ih = 640;
        static const int nw = 432, nh = 576;
        //图像元数据

};

#endif //DEMO_TF_MODEL_H
