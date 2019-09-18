#include "../include/cmpro/rate_method.h"

RateMethod::RateMethod() {
    std::cout << "starting." << std::endl;
    conf.rate_conf();
    conf.process_conf();
    rfleft = 0;
    rftop = 0;
    area_width = conf.WIN_WIDTH;
    area_height = conf.WIN_HEIGHT;
    detector = dlib::get_frontal_face_detector();
    is_loop_continue = true;
    state = 0;
    score = 0;
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

void RateMethod::loop_process(bool is_recorded) {
    try{
        //cap initialization
        cv::VideoCapture cap;
        if(conf.stin == "0"){
            cap.open(0);
        }
        else{
            cap.open("../videoraw/" + conf.stin + ".mp4");
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
            cv::resize(processing_image, processing_image, cv::Size(conf.WIN_WIDTH, conf.WIN_HEIGHT), 0, 0,
                       cv::INTER_LINEAR); //resize image to suitable size
            processing_image = processing_image(cv::Rect(std::max(0L, rfleft), std::max(0L, rftop),
                                                         std::min(conf.WIN_WIDTH - rfleft, area_width),
                                                         std::min(conf.WIN_HEIGHT - rftop, area_height)));
            dlib::cv_image<dlib::bgr_pixel> cimg(processing_image);
            std::vector<dlib::rectangle> faces = detector(cimg);
            dlib::full_object_detection sps;

            clock_weight = clock();
            clock_time = clock();

            sps = pose_model(cimg, faces[0]);

            if (sps.num_parts() >= 68) {
                //if faces points found
                ShapeProcessingClass shape_processing(sps);  //模型处理实例化
                state = 0;

                score += (0.5 - shape_processing.eye_value) * (clock() - clock_weight) / 1000;
                score = std::max(0.0, score);
                score = std::min(conf.SCORE_MAX, score);
                clock_weight = clock();
                //Sub 状态判定
                //fatigue degree judgement
                if (score > conf.SCORE_TOP) {
                    state = 3;
                } else {
                    state = 4;
                }

                if (shape_processing.eye_value < conf.RATE_BOTTOM) {
                    if ((clock() - clock_time) / 1000.0 > conf.PERIOD_AVERAGE) {
                        state = 2;
                    }
                    if ((clock() - clock_time) / 1000.0 > conf.PERIOD_SERIOUS) {
                        state = 1;
                    }
                } else {
                    clock_time = clock();
                }

                //Sub 末端处理(人脸切割)
                //final preperation (face selection)
                rfleft += shape_processing.shape_differ_left - conf.MARGIN_LEFT;
                rftop += shape_processing.shape_differ_top - conf.MARGIN_TOP;
                area_width = shape_processing.shape_width + conf.MARGIN_LEFT + conf.MARGIN_RIGHT;
                area_height = shape_processing.shape_height + conf.MARGIN_TOP + conf.MARGIN_DOWN;

                if(is_recorded){
                    std::ofstream fout;
                    fout.open("../dataout/"+conf.stin+".txt",std::ios::out);
                    fout << 

                }


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
