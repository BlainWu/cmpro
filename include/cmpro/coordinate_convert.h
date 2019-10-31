#ifndef CMPRO_COORDINATECONVERT_H
#define CMPRO_COORDINATECONVERT_H

#include "common_include.h"
#include <cmath>
class CoordinateConvert {
public:
    CoordinateConvert();
private:
    //The convert of (u,v) -> (x,y)
    std::vector<int> single_convert(int u,int v);
    int tu,tv;
    double theta;
};


#endif //CMPRO_COORDINATECONVERT_H
