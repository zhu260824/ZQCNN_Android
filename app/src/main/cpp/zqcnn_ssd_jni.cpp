//
// Created by ZL on 2019-06-13.
//
#include "zqcnn_ssd.h"
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include <imgproc/types_c.h>

#define TAG "ZQCNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)

using namespace std;
using namespace cv;
using namespace ZQ;

static ZQCNNSSD *zqcnnssd;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_zl_demo_ZQCNNSSD_initModelPath(JNIEnv *env, jobject instance, jstring modelPath_) {
    if (NULL == modelPath_) {
        return false;
    }
    //获取MTCNN模型的绝对路径的目录（不是/aaa/bbb.bin这样的路径，是/aaa/)
    const char *modelPath = env->GetStringUTFChars(modelPath_, 0);
    if (NULL == modelPath) {
        return false;
    }
    string tFaceModelDir = modelPath;
    string tLastChar = tFaceModelDir.substr(tFaceModelDir.length() - 1, 1);
    //目录补齐/
    if ("\\" == tLastChar) {
        tFaceModelDir = tFaceModelDir.substr(0, tFaceModelDir.length() - 1) + "/";
    } else if (tLastChar != "/") {
        tFaceModelDir += "/";
    }
    LOGD("init, tFaceModelDir=%s", tFaceModelDir.c_str());
    zqcnnssd = new ZQCNNSSD(tFaceModelDir);
    return true;
}

JNIEXPORT jfloatArray JNICALL
Java_com_zl_demo_ZQCNNSSD_detectFace(JNIEnv *env, jobject instance,jstring imgPath_) {
    const char *imgPath = env->GetStringUTFChars(imgPath_, 0);
    std::vector<ZQ::ZQ_CNN_SSD::BBox> faceInfo = zqcnnssd->detect(imgPath);
    int32_t num_face = static_cast<int32_t>(faceInfo.size());
    LOGD("检测到的人脸数目：%d\n", num_face);
    int out_size = 1 + num_face * 29;
    float *faces = new float[out_size];
    faces[0] = num_face;
    for (int i = 0; i < num_face; i++) {
    }
    jfloatArray tFaces = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(tFaces, 0, out_size, faces);
    return tFaces;
}

}