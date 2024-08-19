//
// Created by João Codagnoni on 04/08/24.
//

#include <cpu.h>
#include "fdtct.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <fstream>

FaceDtct::FaceDtct()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int FaceDtct::load(const char* modeldir, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    faceDtct.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    faceDtct.opt = ncnn::Option();
#if NCNN_VULKAN
    yoloface.opt.use_vulkan_compute = use_gpu;
#endif

    faceDtct.opt.num_threads = ncnn::get_big_cpu_count();
    faceDtct.opt.blob_allocator = &blob_pool_allocator;
    faceDtct.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/fdtct/%s.param", modeldir, modeltype);
    sprintf(modelpath, "%s/fdtct/%s.bin", modeldir, modeltype);
    faceDtct.load_param(parampath);
    faceDtct.load_model(modelpath);

    std::string type(modelpath);
    if(type.find("lite-t") == std::string::npos) lite_t = false;
    else lite_t = true;

    target_size = _target_size;
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

static inline float sigmoid(float x){
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

std::vector<float> softmax(const std::vector<float> & input)
{
    float total = 0.;
    for(auto x : input)
    {
        total += exp(x);
    }
    std::vector<float> result;
    for(auto x : input)
    {
        result.push_back(exp(x) / total);
    }
    return result;
}

bool cmp(Object b1, Object b2) {
    return b1.prob > b2.prob;
}

void nms(std::vector<Object> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

cv::Mat FaceDtct::resize_image(cv::Mat srcimg, int *newh, int *neww, int *padh, int *padw)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = this->inpHeight;
    *neww = this->inpWidth;
    cv::Mat dstimg;
    if (this->keep_ratio && srch != srcw) {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1) {
            *newh = this->inpHeight;
            *neww = int(this->inpWidth / hw_scale);
            resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *padw = int((this->inpWidth - *neww) * 0.5);
            copyMakeBorder(dstimg, dstimg, 0, 0, *padw, this->inpWidth - *neww - *padw, cv::BORDER_CONSTANT, 0);
        }
        else {
            *newh = (int)this->inpHeight * hw_scale;
            *neww = this->inpWidth;
            resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *padh = (int)(this->inpHeight - *newh) * 0.5;
            copyMakeBorder(dstimg, dstimg, *padh, this->inpHeight - *newh - *padh, 0, 0, cv::BORDER_CONSTANT, 0);
        }
    }
    else {
        resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
    }
    return dstimg;
}

void decode(const ncnn::Mat& feat_blob, std::vector<Object> &prebox, float threshold, int stride, int imgh,int imgw, float ratioh, float ratiow, int padh, int padw)
{
    int reg_max         = 16;
    int fea_h 			= feat_blob.h;
    int fea_w 			= feat_blob.w;
    int spacial_size	= fea_w * fea_h;
    float *ptr_b   		= (float*)(feat_blob.data);
    float *ptr_c        = ptr_b + spacial_size * reg_max * 4;
    float *ptr_p        = ptr_c + spacial_size;

    for(int i = 0; i < fea_h; i++)
    {
        for(int j = 0; j < fea_w; j++)
        {
            int index = i * fea_w + j;
            float box_prob 	= sigmoid(ptr_c[index]);
            if(box_prob > threshold)
            {
                float pred_ltrb[4];
                std::vector<float> dfl_value;
                std::vector<float> dfl_softmax;
                dfl_value.resize(reg_max);
                dfl_softmax.resize(reg_max);
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    for(int n=0; n < reg_max; n++){
                        dfl_value[n]=ptr_b[index + (reg_max * k + n) * spacial_size];
                    }

                    dfl_softmax = softmax(dfl_value);
                    for (int l = 0; l < reg_max; l++){
                        dis += l * dfl_softmax[l];
                    }
                    pred_ltrb[k] = dis * stride;
                }

//                float pb_cx = (j + 0.5f) * stride;
//                float pb_cy = (i + 0.5f) * stride;
//
//                float x1 = (pb_cx - pred_ltrb[0]) * ratiow;
//                float y1 = (pb_cy - pred_ltrb[1]) * ratioh;
//                float x2 = (pb_cx + pred_ltrb[2]) * ratiow;
//                float y2 = (pb_cy + pred_ltrb[3]) * ratioh;

                float cx = (j + 0.5f)*stride;
                float cy = (i + 0.5f)*stride;
                float xmin = std::max((cx - pred_ltrb[0] - padw)*ratiow, 0.f);  ///还原回到原图
                float ymin = std::max((cy - pred_ltrb[1] - padh)*ratioh, 0.f);
                float xmax = std::min((cx + pred_ltrb[2] - padw)*ratiow, float(imgw - 1));
                float ymax = std::min((cy + pred_ltrb[3] - padh)*ratioh, float(imgh - 1));

                Object temp_box;
                temp_box.prob 	= box_prob;
                temp_box.label 	= 0;
                temp_box.x1 	= int(xmin);
                temp_box.y1 	= int(ymin);
                temp_box.x2 	= int(xmax);
                temp_box.y2 	= int(ymax);

                prebox.push_back(temp_box);
            }
        }
    }
}

int FaceDtct::detect(const cv::Mat& srcimg, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    cv::Mat dst;
    target_size = 640;
    int img_w = srcimg.cols;
    int img_h = srcimg.rows;

    cv::cvtColor(srcimg, srcimg, cv::COLOR_RGB2BGR, srcimg.channels());

    int newh = 0, neww = 0, padh = 0, padw = 0;
    cv::Mat resized = this->resize_image(srcimg, &newh, &neww, &padh, &padw);
    float ratioh = (float)srcimg.rows / newh;
    float ratiow = (float)srcimg.cols / neww;

    cv::Mat n;
    ncnn::Mat in_pad = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_RGB, resized.cols, resized.rows);

    in_pad.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = faceDtct.create_extractor();
    ex.input("images", in_pad);

    // stride 16
    {
        ncnn::Mat out;
        if(lite_t) ex.extract("884", out);
        else ex.extract("1076", out);
        decode(out, objects, prob_threshold, 16, img_h, img_w, ratioh, ratiow, padh, padw);
    }

    std::sort(objects.begin(), objects.end(), cmp);
    nms(objects, nms_threshold);

    return 0;
}
