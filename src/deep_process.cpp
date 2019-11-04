#include "../include/cmpro/deep_process.h"

DeepProcess::DeepProcess(Configure &config_,dlib::full_object_detection& shape_input):
model_process(config_)
{
    config = config_;
    detected_shape = shape_input;

}

int DeepProcess::deep_cal() {
    std::vector<double> shape_ori,shape_before;
    int result=0;
    for(int i=0;i<68;i++){
        shape_ori.push_back(detected_shape.part(i).x());
        shape_ori.push_back(detected_shape.part(i).y());
    }
    //the process to convert to a sorted input shape
    shape_before = shape_ori;

    result = model_process.model_predict(shape_before);
    if(result >0 && result < 5){
        return result;
    }
    return -1;
}
