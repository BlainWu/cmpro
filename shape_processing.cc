
#include "shape_processing.h"

ShapeProcessingClass::ShapeProcessingClass(dlib::full_object_detection& shape_input) {
    detected_shape = shape_input;
    if((RightEyeValueCalculate()&&LeftEyeValueCalculate()&&MouthValueCalculate()&&AnglePoseDetect())){
        is_updated = true;
        eye_value = (right_eye_value + left_eye_value) * 0.5;
        is_eye_normal = true;
    }
    else{
        std::cerr << "Initializetion Failure Error" << std::endl;
        is_updated = false;
    }
}

void ShapeProcessingClass::DimensionCalculation() {
    int shape_differ_right;
    int shape_differ_down;
    if(is_updated){
        shape_differ_left=(int)detected_shape.part(0).x();
        shape_differ_right=(int)detected_shape.part(16).x();
        shape_differ_top=(int)std::min(detected_shape.part(19).y(),detected_shape.part(24).y());
        shape_differ_down=(int)detected_shape.part(9).y();
        for(unsigned long i=1;i<27;i++){
            shape_differ_left=std::min(shape_differ_left,(int)detected_shape.part(i).x());
            shape_differ_right=std::max(shape_differ_right,(int)detected_shape.part(i).x());
            shape_differ_top=std::min(shape_differ_top,(int)detected_shape.part(i).y());
            shape_differ_down=std::max(shape_differ_down,(int)detected_shape.part(i).y());
        }
        shape_width = shape_differ_right - shape_differ_left;
        shape_height = shape_differ_down - shape_differ_top;
    }
    else{
        std::cerr << "Shape sub not prepared Error" << std::endl;
        return;
    }
}



bool ShapeProcessingClass::RightEyeValueCalculate() {
    if(detected_shape.num_parts()<68){
        std::cerr << "Face detection Error" << std::endl;
        return false;
    }
    long le1=detected_shape.part(39).x()-detected_shape.part(36).x();
    long le2=detected_shape.part(39).y()-detected_shape.part(36).y();
    double le=sqrt(le1*le1+le2*le2);
    long da1=detected_shape.part(41).x()-detected_shape.part(37).x();
    long da2=detected_shape.part(41).y()-detected_shape.part(37).y();
    long db1=detected_shape.part(40).x()-detected_shape.part(38).x();
    long db2=detected_shape.part(40).y()-detected_shape.part(38).y();
    double da=sqrt(da1*da1+da2*da2);
    double db=sqrt(db1*db1+db2*db2);
    if(le<1){
        std::cerr << "eyes detect mistake" << std::endl;
        return false;
    }
    double k=(da+db)/le;
    right_eye_value=k;
    return true;
}

bool ShapeProcessingClass::LeftEyeValueCalculate() {
    if(detected_shape.num_parts()<68){
        std::cerr << "Face detection Error" << std::endl;
        return false;
    }

    long le1=detected_shape.part(45).x()-detected_shape.part(42).x();
    long le2=detected_shape.part(45).y()-detected_shape.part(42).y();
    double le=sqrt(le1*le1+le2*le2);
    long da1=detected_shape.part(47).x()-detected_shape.part(43).x();
    long da2=detected_shape.part(47).y()-detected_shape.part(43).y();
    long db1=detected_shape.part(46).x()-detected_shape.part(44).x();
    long db2=detected_shape.part(46).y()-detected_shape.part(44).y();
    double da=sqrt(da1*da1+da2*da2);
    double db=sqrt(db1*db1+db2*db2);

    if(le<1){
        std::cerr << "eyes detect mistake" << std::endl;
        return false;
    }

    double k=(da+db)/le;
    left_eye_value=k;
    return true;
}

bool ShapeProcessingClass::MouthValueCalculate() {
    if(detected_shape.num_parts()<68){
        std::cerr << "Face detection Error" << std::endl;
        return false;
    }
    long wit1=detected_shape.part(48).x()-detected_shape.part(54).x();
    long wit2=detected_shape.part(48).y()-detected_shape.part(54).y();
    double wit=sqrt(wit1*wit1+wit2*wit2);
    long hei1=detected_shape.part(51).x()-detected_shape.part(57).x();
    long hei2=detected_shape.part(51).y()-detected_shape.part(57).y();
    double hei=sqrt(hei1*hei1+hei2*hei2);
    double m=hei/wit;
    if(m<=0){
        std::cerr << "Mouth detection calculation Error" << std::endl;
        return false;
    }
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