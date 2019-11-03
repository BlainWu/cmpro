#ifndef CMPRO_DEEP_PROCESS_H
#define CMPRO_DEEP_PROCESS_H

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "configure.h"
#include "model.h"
#include "coordinate_convert.h"


class DeepProcess {
public:
    DeepProcess(Configure& config_,dlib::full_object_detection& shape_input);
private:
    Model model_process;
    CoordinateConvert converter;
    Configure config;
    dlib::full_object_detection detected_shape;
};


#endif //CMPRO_DEEP_PROCESS_H
