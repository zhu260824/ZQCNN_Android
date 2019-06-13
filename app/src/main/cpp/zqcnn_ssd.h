//
// Created by ZL on 2019-06-13.
//

#ifndef ZQCNNDEMO_ZQCNN_SSD_H
#define ZQCNNDEMO_ZQCNN_SSD_H

#include "ZQ_CNN_SSD.h"
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_CompileConfig.h"

class ZQCNNSSD {
public:
    ZQCNNSSD(const std::string &model_path);

    std::vector< ZQ::ZQ_CNN_SSD::BBox> detect(const std::string &img_path);

private:
    ZQ::ZQ_CNN_SSD *detector;
};

#endif //ZQCNNDEMO_ZQCNN_SSD_H
