//
// Created by micang on 2021/7/19.
//

#include "tf_model.h"
static const int map_[21][3]={
        {0, 0, 0},{128, 0, 0},{0, 128, 0},{128, 128, 0},{0, 0, 128},{128, 0, 128},{0, 128, 128},
        {128, 128,128},{64, 0, 0},{192, 0, 0},{64, 128, 0},{192, 128, 0},{64, 0, 128},{192, 0, 128},
        {64, 128, 128},{192, 128, 128},{0, 64, 0},{128, 64, 0},{0, 192, 0},{128, 192, 0},{0, 64, 128}
}; //标签
tensorflow::Session *session; //这两个定义无法放在.h的类里，会找不到定义
tensorflow::Tensor pr;
static const std::string input_tensor_name = "input_1:0";
static const std::string output_tensor_name = "main/truediv:0";
// 将Mat地址与Tensor绑定
// https://github.com/tensorflow/tensorflow/issues/8033
void tf_model::CVMat_to_Tensor(cv::Mat& img, tensorflow::Tensor *output_tensor) {
        cv::Mat img_2 = img.clone();
        resize(img_2, img_2, cv::Size(nh, nw), CV_INTER_CUBIC); //576*576
        cv::Rect roi_rect = cv::Rect((size_h - nh) / 2, (size_w - nw) / 2, nh, nw);
        cv::Mat image = cv::Mat(size_w, size_h, CV_8UC3, cv::Scalar(128, 128, 128));
        img_2.copyTo(image(roi_rect));
        //归一化
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        //创建一个指向tensor的内容的指针
        float *p = output_tensor->flat<float>().data();
        cv::Mat tempMat = cv::Mat(size_h, size_w, CV_32FC3, p);
        image.convertTo(tempMat, CV_32FC3);
        tempMat = tempMat / 255.0;
}

void tf_model::loadModel(const std::string& model_path) {
        tensorflow::GraphDef graphdef; //Graph Definition for current model
        tensorflow::SessionOptions sessionOpt;
        sessionOpt.config.mutable_gpu_options()->set_allow_growth(true);
        session = tensorflow::NewSession(sessionOpt);
        tensorflow::Status status_load = ReadBinaryProto(tensorflow::Env::Default(), model_path, &graphdef); //从pb文件中读取图模型;
        tensorflow::Status status_create = session->Create(graphdef);

}

void tf_model::sess(cv::Mat& img) {
        //创建一个tensor作为输入网络的接口
        tensorflow::Tensor resized_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, size_h, size_w, 3}));
        /*--------------------------------创建session------------------------------*/
        //将Opencv的Mat格式的图片存入tensor
        CVMat_to_Tensor(img, &resized_tensor);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status_run = session->Run({{input_tensor_name, resized_tensor}}, {output_tensor_name}, {}, &outputs);
        pr = outputs.at(0);
}


void tf_model::ge_label(cv::Mat& imSeg) {
        cv::Mat label = cv::Mat(nw, nh, CV_8UC3);
        int output_dim = pr.shape().dim_size(3);  // Get the class_num from -1st dimension
        float *q = pr.flat<float>().data();
        cv::Mat conf(size_h, size_w, CV_32FC(21), q);
        for (int i = (size_w-nw)/2, i1 = 0; i < (size_w-nw)/2+nw; ++i, ++i1) {
            cv::Vec<float, 21> *p1 = conf.ptr<cv::Vec<float, 21>>(i);
            cv::Vec3b *p2 = label.ptr<cv::Vec3b>(i1);
            for (int j = (size_h-nh)/2, j1 = 0; j < (size_h-nh)/2+nh; ++j, ++j1) {
                int index = 0;
                float swap = 0;
                for (int c = 0; c < output_dim; ++c)//对标签进行排序，选几率最大的一个作为结果
                    {
                    if (swap < p1[j][c])
                    {
                        swap = p1[j][c];
                        index = c;//index为标签
                        if(swap > 0.5f) //置信度大于0.5必然是标签
                            break;
                    }
                    }
                p2[j1][0] = map_[index][2];
                p2[j1][1] = map_[index][1];
                p2[j1][2] = map_[index][0];
            }
        }
        cv::resize(label, imSeg, cv::Size(ih, iw), cv::INTER_NEAREST);
}
