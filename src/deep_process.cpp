#include "../include/cmpro/deep_process.h"

DeepProcess::DeepProcess() :config(),converter(),seqconverter(config){
    std::cout << "starting." << std::endl;
    rfleft = 0;
    rftop = 0;
    area_width = config.WIN_WIDTH;
    area_height = config.WIN_HEIGHT;
    detector = dlib::get_frontal_face_detector();
    is_loop_continue = true;
    state = 0;
    score = 0;
    is_updated =  true;
    try{
        dlib::deserialize("../resource/shape_predictor_68_face_landmarks.dat") >> pose_model;
    }
    catch (dlib::serialization_error& e)
    {
        std::cout << std::endl << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}

void DeepProcess::loop_process() {
    try{
        //cap initialization
        cv::VideoCapture cap;
        if(config.stin == "0"){
            cap.open(0);
        }
        else{
            cap.open("../videoraw/" + config.stin + ".mp4");
        }

        if (!cap.isOpened()) {
            std::cerr << "Unable to connect to camera" << std::endl;
            return;
        }
        //ShowingWindow showing_window(&cap);

        std::cout << "initialized." << std::endl;

        while(cv::waitKey(10)!='q' && is_loop_continue) {
            cap >> showing_image;
            cv::Mat processing_image;
            processing_image = showing_image.clone();
            cv::resize(processing_image, processing_image, cv::Size(config.WIN_WIDTH, config.WIN_HEIGHT), 0, 0,
                       cv::INTER_LINEAR); //resize image to suitable size
            processing_image = processing_image(cv::Rect(std::max(0L, rfleft), std::max(0L, rftop),
                                                         std::min(config.WIN_WIDTH - rfleft, area_width),
                                                         std::min(config.WIN_HEIGHT - rftop, area_height)));
            dlib::cv_image<dlib::bgr_pixel> cimg(processing_image);
            std::vector<dlib::rectangle> faces = detector(cimg);
            dlib::full_object_detection sps;

            clock_weight = clock();
            clock_time = clock();


            if (!faces.empty()) {
                sps = pose_model(cimg, faces[0]);
                //if faces points found





                //state = shape_processing.deep_cal();

                if(state == 1){
                    score += 250;//Serious
                }
                if(state == 2){
                    score += 200;//Average
                }
                if(state ==3){
                    score += 75;//Slight
                }
                if(state ==4){
                    score -= 50;//Normal
                }

                score = std::max(0.0, score);
                score = std::min(config.SCORE_MAX, score);
                clock_weight = clock();


                //Sub 末端处理(人脸切割)
                //final preperation (face selection)
                rfleft += shape_differ_left - config.MARGIN_LEFT;
                rftop += shape_differ_top - config.MARGIN_TOP;
                area_width = shape_width + config.MARGIN_LEFT + config.MARGIN_RIGHT;
                area_height = shape_height + config.MARGIN_TOP + config.MARGIN_DOWN;

                putText(showing_image, show_msg[state], cv::Point(10, 60), cv::QT_FONT_NORMAL, 1, cvScalar(0, 0, 255),
                        1, 1);
                std::cout << ctmsg[state] << std::endl;

                imshow("cap", showing_image);
            }
        }

    }
    catch (dlib::serialization_error& e)
    {
        std::cout << std::endl << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}



int DeepProcess::deep_cal() {
    std::vector<int> shape_ori;
    std::vector<double> shape_before;
    int result=0;
    for(int i=0;i<68;i++){
        shape_ori.push_back(detected_shape.part(i).x());
        shape_ori.push_back(detected_shape.part(i).y());
    }
    //the process to convert to a sorted input shape
    shape_before = converter.multi_convert(shape_ori);

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