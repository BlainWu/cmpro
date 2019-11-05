#ifndef CMPRO_SEQUENCE_CONVERT_H
#define CMPRO_SEQUENCE_CONVERT_H

#include "common_include.h"
#include "configure.h"

class SequenceConvert {
public:
    SequenceConvert(Configure& config);
    std::vector<std::vector<double> > newdata
    (std::vector<double> new_value_,double new_dur_);
private:
    double duration;
    std::vector<double> former;

    int shape_len;

    double interval;

    double acu_dur;
};


#endif //CMPRO_SEQUENCE_CONVERT_H
