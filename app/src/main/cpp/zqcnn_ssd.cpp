//
// Created by ZL on 2019-06-13.
//

#include "zqcnn_ssd.h"
#include <android/log.h>

#define TAG "ZQCNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)

using namespace ZQ;
using namespace std;
using namespace cv;

ZQCNNSSD::ZQCNNSSD(const std::string &model_path) {
    string proto_file = model_path + "libfacedetection.zqparams";
    string model_file = model_path + "libfacedetection.nchwbin";
    detector=new ZQ_CNN_SSD();
    bool init = detector->Init(proto_file, model_file, "detection_out");
    LOGD("MTCNN_init：%d\n", init);
}

std::vector<ZQ::ZQ_CNN_SSD::BBox> ZQCNNSSD::detect(const std::string &img_path) {
    std::vector<ZQ_CNN_SSD::BBox> output;
    Mat img = cv::imread(img_path, 1);
    if (img.empty()) {
        return output;
    }
    Mat img0;
    cv::cvtColor(img, img0, CV_BGR2RGB);
    int out_iter = 1;
    int iters = 1;
    const float kScoreThreshold = 0.3f;
    for (int out_it = 0; out_it < out_iter; out_it++) {
        double t1 = omp_get_wtime();
        for (int it = 0; it < iters; it++) {
            if (!detector->Detect(output, img0.data, img0.cols, img0.rows, img0.step[0],
                                  kScoreThreshold, false)) {
                return output;
            }
        }
        double t2 = omp_get_wtime();
        LOGD("[%d] times cost %.3f s, 1 iter cost %.3f ms\n", iters, t2 - t1,1000 * (t2 - t1) / iters);
    }
    return output;
}