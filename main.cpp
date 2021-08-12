#include "tf_model.h"
using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
    string model_path="./de-mobilenet.pb";
    string image_path="./8.png"; //对应的是imRGB的路径
    Mat imRGB=imread(image_path,IMREAD_UNCHANGED);
    tf_model *tfModel;
    cv::Mat imSeg(tfModel->nw,tfModel->nh,CV_8UC3);
    tfModel->loadModel(model_path);//载入模型
    tfModel->sess(imRGB);
    tfModel->ge_label(imSeg);
    imshow("label",imSeg);
    waitKey(0==27);
    return 0;
}
