#ifndef CMPRO_RATE_METHOD_H
#define CMPRO_RATE_METHOD_H

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <pthread.h>

#include "configure.h"

class RateMethod {
public:
    RateMethod();
private:
    Configure conf;

    dlib::frontal_face_detector detector;
    dlib::shape_predictor pose_model;
    cv::Mat showing_image;

    //    //if value calculation loop continue
    //    bool is_loop_continue ;
    //    //state 疲劳状态:0未识别，1重度，2中度，3轻度，4正常，5警告广播
    //    //state 0~Not Leveled; 1~Degree-3; 2~Degree-2; 3~Degree-1; 4~Degree-0(Normal); 5~Alarming Signal ..
    int state;

    //position and size of the selected area
    long rfleft, rftop, area_width, area_height;

    double score;
    clock_t clock_weight ;
    clock_t clock_time ;

    //各种状态的显示消息
    const std::vector<std::string> ctmsg = {"未匹配","重度3","中度2","轻度1","正常0","状态异常"};
    const std::vector<std::string> show_msg = {" No-Matching","degree-3","degree-2 ","degree-1","degree-0","Alarming"};
};


#endif //CMPRO_RATE_METHOD_H
