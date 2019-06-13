//
// Created by ZL on 2019-06-13.
//

#ifndef ZQCNNDEMO_ZQCNN_MTCNN_NCHWC4_H
#define ZQCNNDEMO_ZQCNN_MTCNN_NCHWC4_H

#include "ZQ_CNN_Net_NCHWC.h"
#include "ZQ_CNN_MTCNN_NCHWC.h"
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_CompileConfig.h"
#include <cblas.h>

class MTCNNNCHWC {
public:
    MTCNNNCHWC(const std::string &model_path);

    std::vector<ZQ::ZQ_CNN_BBox> detect(const std::string &img_path);


    std::vector<ZQ::ZQ_CNN_BBox> detectMat(cv::Mat &image0);

private:
    ZQ::ZQ_CNN_MTCNN_NCHWC *mtcnn;
    int thread_num = 1;
};

#endif //ZQCNNDEMO_ZQCNN_MTCNN_NCHWC4_H
