//
// Created by João Codagnoni on 04/08/24.
//

#ifndef TT_FACE_API_FDTCT_H
#define TT_FACE_API_FDTCT_H

#include <opencv2/core.hpp>

#include <net.h>
struct Landmarks
{
    float x;
    float y;
    float score;
};
struct Object
{
    float x1;
    float y1;
    float x2;
    float y2;
    int label;
    float prob;
};


class FaceDtct
{
public:
    FaceDtct();

    int load(const char* modeldir, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& srcimg, std::vector<Object>& objects, float prob_threshold = 0.25f, float nms_threshold = 0.45f);


private:
    const int inpWidth = 640;
    const int inpHeight = 640;
    const bool keep_ratio = true;

    cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *padh, int *padw);

    ncnn::Net faceDtct;

    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    bool lite_t = false;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif //TT_FACE_API_FDTCT_H
