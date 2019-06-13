//
// Created by ZL on 2019-06-13.
//

#include <imgproc/types_c.h>
#include "zqcnn_mtcnn.h"
#include <android/log.h>

#define TAG "ZQCNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)

using namespace ZQ;
using namespace std;
using namespace cv;

ZQMTCNN::ZQMTCNN(const std::string &model_path) {
    mtcnn = new ZQ_CNN_MTCNN();
    bool init = mtcnn->Init(model_path + "det1-dw20-fast.zqparams",
                            model_path + "det1-dw20-fast.nchwbin",
                            model_path + "det2-dw24-fast.zqparams",
                            model_path + "det2-dw24-fast.nchwbin",
                            model_path + "det3-dw48-fast.zqparams",
                            model_path + "det3-dw48-fast.nchwbin",
                            thread_num, false,
                            model_path + "det4-dw48-v2s.zqparams",
                            model_path + "det4-dw48-v2s.nchwbin");
    /*bool init = mtcnn->Init(model_path + "det1.zqparams",
                            model_path + "det1_bgr.nchwbin",
                            model_path + "det2.zqparams",
                            model_path + "det2_bgr.nchwbin",
                            model_path + "det3.zqparams",
                            model_path + "det3_bgr.nchwbin",
                            thread_num);*/
    LOGD("MTCNN_init：%d\n", init);
}

std::vector<ZQ::ZQ_CNN_BBox> ZQMTCNN::detect(const std::string &img_path) {
    std::vector<ZQ_CNN_BBox> thirdBbox;
    Mat image0 = cv::imread(img_path, 1);
    if (image0.empty()) {
        return thirdBbox;
    }
    if (image0.channels() == 1)
        cv::cvtColor(image0, image0, CV_GRAY2BGR);
    bool run_blur = true;
    int kernel_size = 3, sigma = 2;
    if (image0.cols * image0.rows >= 2500 * 1600) {
        run_blur = false;
        kernel_size = 5;
        sigma = 3;
    } else if (image0.cols * image0.rows >= 1920 * 1080) {
        run_blur = false;
        kernel_size = 3;
        sigma = 2;
    } else {
        run_blur = false;
    }
    if (run_blur) {
        cv::Mat blur_image0;
        int nBlurIters = 1000;
        double t00 = omp_get_wtime();
        for (int i = 0; i < nBlurIters; i++)
            cv::GaussianBlur(image0, blur_image0, cv::Size(kernel_size, kernel_size), sigma, sigma);
        double t01 = omp_get_wtime();
        printf("[%d] blur cost %.3f secs, 1 blur costs %.3f ms\n", nBlurIters, t01 - t00,
               1000 * (t01 - t00) / nBlurIters);
        cv::GaussianBlur(image0, image0, cv::Size(kernel_size, kernel_size), sigma, sigma);
    }
    mtcnn->SetPara(image0.cols, image0.rows, 40, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4,
                   false);

//    mtcnn->SetPara(image0.cols, image0.rows, 20, 0.6, 0.7, 0.7, 0.4, 0.5, 0.5, 0.709, 4, 12, 2,
//                   false);
    int iters = 5;
    double t1 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        if (!mtcnn->Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox)) {
            continue;
        }

    }
    double t2 = omp_get_wtime();
    LOGD("total %.3f s / %d = %.3f ms\n", t2 - t1, iters, 1000 * (t2 - t1) / iters);
    return thirdBbox;
}