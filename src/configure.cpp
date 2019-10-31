//
// Created by kanae on 2019-09-10.
//

#include "../include/cmpro/configure.h"

Configure::Configure():
config_process("../configure/process_configure.yaml",cv::FileStorage::READ)
{
    WIN_WIDTH = (int)config_process["window_width"];
    WIN_HEIGHT = (int)config_process["window_height"];
    MARGIN_LEFT = (int)config_process["margin_left"];
    MARGIN_TOP = (int)config_process["margin_top"];
    MARGIN_RIGHT = (int)config_process["margin_right"];
    MARGIN_DOWN = (int)config_process["margin_down"];
    stin = (std::string)config_process["video_path"];

    RATE_TOP = (double)config_process["rate_top"];
    RATE_BOTTOM = (double)config_process["rate_bottom"];
    MOUTH_TOP = (double)config_process["mouth_top"];
    ANGLE_JUDGEMENT = (double)config_process["angle_judgement"];
    SCORE_TOP = (double)config_process["score_top"];
    SCORE_MAX = (double)config_process["score_max"];
    PERIOD_AVERAGE = (double)config_process["period_average"];
    PERIOD_SERIOUS = (double)config_process["period_serious"];
}
