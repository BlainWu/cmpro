//
// Created by kanae on 2019-09-10.
//

#include "../include/cmpro/configure.h"

void Configure::rate_conf() {
    std::ifstream fin;
    fin.open("../configure/rate_configure.dat");
    std::vector<double> din;
    din.resize(din_num);
    for(int i=0;i<din_num;++i){
        fin >> din[i];
    }
    RATE_TOP = din[0];
    RATE_BOTTOM = din[1];
    ANGLE_JUDGEMENT = din[2];
    MOUTH_TOP = din[3];
    SCORE_TOP = din[4];
    SCORE_MAX = din[5];
    PERIOD_AVERAGE = din[6];
    PERIOD_SERIOUS = din[7];
    fin.close();
}

void Configure::process_conf() {
    std::ifstream fin;
    fin.open("../configure/process_configure.dat");
    std::vector<int> pin;
    pin.resize(pin_num);
    for(int i=0;i<pin_num;++i){
        fin >> pin[i];
    }
    fin >> stin;
    WIN_WIDTH = pin[0];
    WIN_HEIGHT = pin[1];
    MARGIN_LEFT = pin[2];
    MARGIN_TOP = pin[3];
    MARGIN_RIGHT = pin[4];
    MARGIN_DOWN = pin[5];
    fin.close();
}
