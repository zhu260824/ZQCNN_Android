//
// Created by ZL on 2019-06-13.
//

#ifndef ZQCNNDEMO_ZQCNN_MTCNN_H
#define ZQCNNDEMO_ZQCNN_MTCNN_H

#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_MTCNN_old.h"
#include "ZQ_CNN_MTCNN.h"
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_CompileConfig.h"


class ZQMTCNN {
public:
    ZQMTCNN(const std::string &model_path);

    std::vector<ZQ::ZQ_CNN_BBox> detect(const std::string &img_path);

private:
    ZQ::ZQ_CNN_MTCNN *mtcnn;
    int thread_num = 1;
};

#endif //ZQCNNDEMO_ZQCNN_MTCNN_H
