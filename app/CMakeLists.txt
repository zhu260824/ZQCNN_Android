cmake_minimum_required(VERSION 3.4.1)

include_directories(src/main/cpp/opencv2
        src/main/cpp/opencv
        src/main/cpp/include
        src/main/cpp/)

file(GLOB MTCNN_SRC src/main/cpp/*.h
        src/main/cpp/*.cpp
        src/main/cpp/zqcnn_mtcnn.cpp
        src/main/cpp/zqcnn_jni.cpp
        src/main/cpp/zqcnn_ssd.cpp
        src/main/cpp/zqcnn_ssd_jni.cpp
        src/main/cpp/zqcnn_mtcnn_nchwc4.cpp
        src/main/cpp/zqcnn_mtcnn_nchwc4_jni.cpp)

set(MTCNN_COMPILE_CODE ${MTCNN_SRC})

add_library(libopencv_java3 SHARED IMPORTED)

set_target_properties(libopencv_java3 PROPERTIES  IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so)

add_library(libZQCNN STATIC IMPORTED)
set_target_properties(libZQCNN
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libZQCNN.a)

add_library(libZQ_GEMM STATIC IMPORTED)
set_target_properties(libZQ_GEMM
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libZQ_GEMM.a)


add_library(libopenblas STATIC IMPORTED)
set_target_properties(libopenblas
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopenblas.a)

add_library(mtcnn SHARED ${MTCNN_COMPILE_CODE})

find_library(log-lib log)

target_link_libraries(
        mtcnn
        libopencv_java3
        libZQCNN
        libZQ_GEMM
        libopenblas
        jnigraphics
        z
        ${log-lib})