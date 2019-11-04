#include "../include/cmpro/deep_process.h"

DeepProcess::DeepProcess(Configure &config_,dlib::full_object_detection& shape_input):
model_process(config_),converter()
{
    config = config_;
    detected_shape = shape_input;
    is_updated =  true;
    DimensionCalculation();
}

int DeepProcess::deep_cal() {
    std::vector<double> shape_ori,shape_before;
    int result=0;
    for(int i=0;i<68;i++){
        shape_ori.push_back(detected_shape.part(i).x());
        shape_ori.push_back(detected_shape.part(i).y());
    }
    //the process to convert to a sorted input shape
    shape_before = converter.multi_convert(shape_ori);

    result = model_process.model_predict(shape_before);
    if(result >0 && result < 5){
        return result;
    }
    return -1;
}

void DeepProcess::DimensionCalculation() {
    long shape_differ_right=0;
    long shape_differ_down=0;
    if(is_updated){
        shape_differ_left=detected_shape.part(0).x();
        shape_differ_right=detected_shape.part(16).x();
        shape_differ_top=std::min(detected_shape.part(19).y(),detected_shape.part(24).y());
        shape_differ_down=detected_shape.part(9).y();
        for(unsigned long i=1;i<27;i++){
            shape_differ_left=std::min(shape_differ_left,detected_shape.part(i).x());
            shape_differ_right=std::max(shape_differ_right,detected_shape.part(i).x());
            shape_differ_top=std::min(shape_differ_top,detected_shape.part(i).y());
            shape_differ_down=std::max(shape_differ_down,detected_shape.part(i).y());
        }
        shape_width = shape_differ_right - shape_differ_left;
        shape_height = shape_differ_down - shape_differ_top;
    }
    else{
        std::cerr << "Shape sub not prepared Error" << std::endl;
        return;
    }
}