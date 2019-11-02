#ifndef CMPRO_COORDINATECONVERT_H
#define CMPRO_COORDINATECONVERT_H

#include "common_include.h"
#include <cmath>
class CoordinateConvert {
public:
    CoordinateConvert();
    std::vector<int> multi_convert(std::vector<int> uvdata);
private:
    bool is_determined;
    //The convert of (u,v) -> (x,y)
    std::vector<double> single_convert(int u,int v);
    //Coorinate Convert Paras input
    void ccpin(int tu_,int tv_,int deltau_,int deltav_);

    int x_int,y_int;
    double x_double,y_double;

    int tu,tv,deltau,deltav;
    double theta;
};


#endif //CMPRO_COORDINATECONVERT_H
