#include "../include/cmpro/deep_process.h"

DeepProcess::DeepProcess(Configure &config_,dlib::full_object_detection& shape_input):
model_process(config_)
{
    config = config_;
    detected_shape = shape_input;
}
