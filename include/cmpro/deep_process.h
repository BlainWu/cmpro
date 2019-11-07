#ifndef CMPRO_DEEP_PROCESS_H
#define CMPRO_DEEP_PROCESS_H

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
//#include "tensorflow/core/public/session.h"
//#include "tensorflow/core/platform/env.h"

#include "configure.h"
#include "coordinate_convert.h"
#include "sequence_convert.h"

class DeepProcess {
public:
    DeepProcess();
    int deep_cal();

    void loop_process();
    bool is_loop_continue;



    long shape_differ_left;
    long shape_differ_top;
    long shape_width;
    long shape_height;

    bool is_updated;
private:
    void DimensionCalculation();
    CoordinateConvert converter;
    SequenceConvert seqconverter;
    Configure config;
    dlib::full_object_detection detected_shape;

    dlib::frontal_face_detector detector;
    dlib::shape_predictor pose_model;
    cv::Mat showing_image;

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


#endif //CMPRO_DEEP_PROCESS_H
