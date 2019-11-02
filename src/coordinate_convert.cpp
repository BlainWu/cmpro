#include "../include/cmpro/coordinate_convert.h"


CoordinateConvert::CoordinateConvert() {
    is_determined = false;
}

std::vector<double> CoordinateConvert::single_convert(int u, int v) {
    if(!is_determined){
        return {0,0};
    }
    double beta = atan((v-tv)/(u-tu))-theta;
    x_double = sqrt((u-tu)*(u-tu)+(v-tv)*(v-tv))*cos(beta);
    y_double = sqrt((u-tu)*(u-tu)+(v-tv)*(v-tv))*sin(beta);
    x_int = (int)x_double;
    y_int = (int)y_double;
    x_double = x_double*10/deltau;
    y_double = y_double*10/deltau;
    return {x_double,y_double};
}

void CoordinateConvert::ccpin(int tu_, int tv_, int deltau_, int deltav_) {
    tu=tu_;
    tv=tv_;
    deltau=deltau_;
    deltav=deltav_;
    theta = atan(deltav/deltau);
    is_determined = true;
}

std::vector<int> CoordinateConvert::multi_convert(std::vector<int> uvdata) {
    std::vector<int> ret;
    if(!is_determined){
        return {0};
    }
    for(int i=0;i<68;i++){
        single_convert(uvdata[2*i],uvdata[2*i+1]);
        ret.push_back(x_double);
        ret.push_back(y_double);
    }
    return ret;
}

