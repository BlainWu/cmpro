
#include "../include/cmpro/shape_processing.h"

ShapeProcessingClass::ShapeProcessingClass
()
{



}

void ShapeProcessingClass::image_select(dlib::full_object_detection& shape_input) {
    detected_shape = shape_input;
    DimensionCalculation();
}

void ShapeProcessingClass::value_cal(std::vector<double>& vector_input_) {
    vector_input = vector_input_;
    if((RightEyeValueCalculate()&&LeftEyeValueCalculate()&&MouthValueCalculate())){
        eye_value = (right_eye_value + left_eye_value) * 0.5;
        is_eye_normal = true;
        DimensionCalculation();
    }
    else{
        std::cerr << "Initializetion Failure Error" << std::endl;

    }
}



void ShapeProcessingClass::DimensionCalculation() {
    long shape_differ_right=0;
    long shape_differ_down=0;

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
    return;
}



bool ShapeProcessingClass::RightEyeValueCalculate() {
    if(vector_input.size()<132){
        std::cerr << "Face detection Error" << std::endl;
        return false;
    }
    double da=vector_input[(42-2)*2+1]-vector_input[(38-2)*2+1];
    double db=vector_input[(41-2)*2+1]-vector_input[(39-2)*2+1];
    double le=vector_input[(40-2)*2]-vector_input[(37-2)*2];
    double k=(da+db)/le;
    right_eye_value=k;
    return true;
}

bool ShapeProcessingClass::LeftEyeValueCalculate() {
    if(vector_input.size()<132){
        std::cerr << "Face detection Error" << std::endl;
        return false;
    }
    double da=vector_input[(48-2)*2+1]-vector_input[(44-2)*2+1];
    double db=vector_input[(47-2)*2+1]-vector_input[(45-2)*2+1];
    double le=vector_input[(46-2)*2]-vector_input[(43-2)*2];
    double k=(da+db)/le;
    left_eye_value=k;
    return true;
}

bool ShapeProcessingClass::MouthValueCalculate() {
    if(vector_input.size()<132){
        std::cerr << "Face detection Error" << std::endl;
        return false;
    }
    double hei=vector_input[(52-2)*2+1]-vector_input[(58-2)*2+1];
    double wit=vector_input[(55-2)*2]-vector_input[(49-2)*2];
    double k=hei/wit;
    mouth_value = k;
    return true;
}

bool ShapeProcessingClass::AnglePoseDetect() {
    cv::Mat rotation_vec;                           //3 x 1
    cv::Mat rotation_mat;                           //3 x 3 R
    cv::Mat translation_vec;                        //3 x 1 T
    cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
    cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);
    //temp buf for decomposeProjectionMatrix()
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);
    cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
    cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

    std::vector<cv::Point2d> reprojectdst;
    reprojectdst.resize(8);

    std::vector<cv::Point2d> image_pts;
    image_pts.push_back(cv::Point2d(detected_shape.part(17).x(), detected_shape.part(17).y())); //#17 left brow left corner
    image_pts.push_back(cv::Point2d(detected_shape.part(21).x(), detected_shape.part(21).y())); //#21 left brow right corner
    image_pts.push_back(cv::Point2d(detected_shape.part(22).x(), detected_shape.part(22).y())); //#22 right brow left corner
    image_pts.push_back(cv::Point2d(detected_shape.part(26).x(), detected_shape.part(26).y())); //#26 right brow right corner
    image_pts.push_back(cv::Point2d(detected_shape.part(36).x(), detected_shape.part(36).y())); //#36 left eye left corner
    image_pts.push_back(cv::Point2d(detected_shape.part(39).x(), detected_shape.part(39).y())); //#39 left eye right corner
    image_pts.push_back(cv::Point2d(detected_shape.part(42).x(), detected_shape.part(42).y())); //#42 right eye left corner
    image_pts.push_back(cv::Point2d(detected_shape.part(45).x(), detected_shape.part(45).y())); //#45 right eye right corner
    image_pts.push_back(cv::Point2d(detected_shape.part(31).x(), detected_shape.part(31).y())); //#31 nose left corner
    image_pts.push_back(cv::Point2d(detected_shape.part(35).x(), detected_shape.part(35).y())); //#35 nose right corner
    image_pts.push_back(cv::Point2d(detected_shape.part(48).x(), detected_shape.part(48).y())); //#48 mouth left corner
    image_pts.push_back(cv::Point2d(detected_shape.part(54).x(), detected_shape.part(54).y())); //#54 mouth right corner
    image_pts.push_back(cv::Point2d(detected_shape.part(57).x(), detected_shape.part(57).y())); //#57 mouth central bottom corner
    image_pts.push_back(cv::Point2d(detected_shape.part(8).x(), detected_shape.part(8).y()));   //#8 chin corner
    //calculate
    cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);
    //reproject
    cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);
    cv::Rodrigues(rotation_vec, rotation_mat);
    cv::hconcat(rotation_mat, translation_vec, pose_mat);
    cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

    angle_X = euler_angle.at<double>(0);
    angle_Y = euler_angle.at<double>(1);
    angle_Z = euler_angle.at<double>(2);
    return true;
}



