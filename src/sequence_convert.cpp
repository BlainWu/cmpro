#include "../include/cmpro/sequence_convert.h"

SequenceConvert::SequenceConvert():config() {
    shape_len = config.TENSOR_SHAPE;
    interval = config.INTERVAL;
    former.resize(shape_len);
    duration = 0;
}

std::vector<std::vector<double> > SequenceConvert::newdata
(std::vector<double> new_value_,double new_dur_)
{
    double new_dur = new_dur_;
    std::vector<std::vector<double>> ret;
    if(new_dur_>100*interval){
        return ret;
    }
    while(duration+new_dur>interval){
        std::vector<double> value_now;
        value_now.resize(shape_len);
        for(int i=0;i<shape_len;i++){
            value_now[i]=(duration*former[i]+(interval-duration)*new_value_[i])/interval;
        }
        new_dur -= interval+duration;
        duration = 0;
        former.clear();
        ret.push_back(value_now);
    }

    duration += new_dur;
    for(int i=0;i<shape_len;i++){
        former[i]=(duration*former[i]+new_dur*new_value_[i])/(duration+new_dur);
    }

    return ret;
}

