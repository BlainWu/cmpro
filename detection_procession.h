#ifndef CMPRO_DETECTION_PROCESSION_H
#define CMPRO_DETECTION_PROCESSION_H

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <pthread.h>

#include "shape_processing.h"
#include "showing_window.h"


class DetectionProcession {
public:
    DetectionProcession();

    void loop_processing();

    bool is_loop_continue ;
private:

    dlib::frontal_face_detector detector;
    dlib::shape_predictor pose_model;
    //-----------------------------------------------------------------
    void handle(void);
    static void* handle_pth(void*);
    void lanch_handle();

    //state 疲劳状态:0未识别，1重度，2中度，3轻度，4正常，5警告广播
    int state;

    cv::Mat showing_image;

    long rfleft, rftop, area_width, area_height;


    double score;

    unsigned long period_weight;

    clock_t clock_weight ;
    clock_t clock_time ;


    //-----------------------------------------------------------------
    double RATE_TOP;
    double RATE_BOTTOM;
    double ANGLE_JUDGEMENT;
    double MOUTH_TOP;

    double PERIOD_AVERAGE;
    double PERIOD_SERIOUS;

    double SCORE_TOP;
    double SCORE_MAX;

    //框定人脸区域上下左右预留出来的误差范围
    int MARGIN_LEFT ;
    int MARGIN_RIGHT ;
    int MARGIN_TOP ;
    int MARGIN_DOWN ;

    //识别的画面初次压缩的最终大小
    //注意，和摄像头调成同比例的，建议在200以上，350以下
    int WIN_WIDTH ;
    int WIN_HEIGHT ;

    bool IS_DEBUG_WIN_ON ;      //是否显示Debug窗体
    bool IS_DEBUG_WIN_DRAWED ;  //是否在Debug窗体上显示特征点（需要Debug窗体显示）
    bool IS_SHOWN_WIN_ON ;       //是否显示展示窗体

    //各种状态的显示消息
    const std::vector<std::string> ctmsg = {" No matching","3 重度","2 中度","1 轻度","0 正常","状态异常"};
    const std::vector<std::string> shownum = {" No matching","degree-3","degree-2 ","degree-1","degree-0","Alarming"};


};

#endif //CMPRO_DETECTION_PROCESSION_H

