//
// Created by kanae on 2019-09-10.
//

#ifndef CMPRO_CONFIGURE_H
#define CMPRO_CONFIGURE_H

#include "common_include.h"
#include <opencv2/core.hpp>
class Configure {
public:
    Configure();
    std::string stin;

    double RATE_TOP;
    double RATE_BOTTOM;
    double ANGLE_JUDGEMENT;
    double MOUTH_TOP;

    double SCORE_TOP;
    double SCORE_MAX;

    double PERIOD_AVERAGE;
    double PERIOD_SERIOUS;

//框定人脸区域上下左右预留出来的误差范围
//enlarged length in face selecting process
    int MARGIN_LEFT ;
    int MARGIN_RIGHT ;
    int MARGIN_TOP ;
    int MARGIN_DOWN ;

//识别的画面初次压缩的最终大小
//注意，和摄像头调成同比例的，建议在200以上，350以下
//first compressed rate(size)
    int WIN_WIDTH ;
    int WIN_HEIGHT ;

//Deep Method
    std::string MODEL_NAME;
private:
    cv::FileStorage config_process;

};


#endif //CMPRO_CONFIGURE_H
