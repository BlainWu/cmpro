#include "detection_procession.h"

DetectionProcession::DetectionProcession() {
    const int pin_num = 6;
    const int din_num = 8;

    std::ifstream fin;
    fin.open("../process_configure.dat");
    std::vector<int> pin;
    pin.resize(pin_num);
    for(int i=0;i<pin_num;++i){
        fin >> pin[i];
    }
    WIN_WIDTH = pin[0];
    WIN_HEIGHT = pin[1];
    MARGIN_LEFT = pin[2];
    MARGIN_TOP = pin[3];
    MARGIN_RIGHT = pin[4];
    MARGIN_DOWN = pin[5];
    fin.close();
    fin.open("../judgement_configure.dat");
    std::vector<double> din;
    din.resize(din_num);
    for(int i=0;i<din_num;++i){
        fin >> din[i];
    }
    RATE_TOP = din[0];
    RATE_BOTTOM = din[1];
    ANGLE_JUDGEMENT = din[2];
    MOUTH_TOP = din[3];
    PERIOD_AVERAGE = din[4];
    PERIOD_SERIOUS = din[5];
    SCORE_TOP = din[6];
    SCORE_MAX = din[7];
    fin.close();

    score = 0;
    state = 0;
    is_loop_continue = true;

    rfleft = 0;
    rftop = 0;
    area_width = WIN_WIDTH;
    area_height = WIN_HEIGHT;
}


void DetectionProcession::loop_processing() {
    try{
        std::cout << "starting." << std::endl;

        detector = dlib::get_frontal_face_detector();

        dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;
        cv::VideoCapture cap("../tst2.mp4");
        if (!cap.isOpened()) {
            std::cerr << "Unable to connect to camera" << std::endl;
            return;
        }
        //ShowingWindow showing_window(&cap);

        std::cout << "initialized." << std::endl;

        while(cv::waitKey(10)!='q' && is_loop_continue){

            cap >> showing_image;

            lanch_handle();

            putText(showing_image,shownum[state],cv::Point(10,60), CV_FONT_NORMAL, 1, cvScalar(0,0,255),1,1);
            //std::cout << ctmsg[state] << std::endl;

            imshow("cap",showing_image);
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
    return;
}

void DetectionProcession::handle() {
    cv:: Mat processing_image;
    processing_image = showing_image.clone();
    cv::resize(processing_image, processing_image, cv::Size(WIN_WIDTH,WIN_HEIGHT), 0, 0, CV_INTER_LINEAR); //resize image to suitable size
    processing_image=processing_image(cv::Rect(std::max(0L,rfleft), std::max(0L,rftop),
                                               std::min(WIN_WIDTH-rfleft, area_width), std::min(WIN_HEIGHT-rftop, area_height)));
    dlib::cv_image<dlib::bgr_pixel> cimg(processing_image);
    std::vector<dlib::rectangle> faces = detector(cimg);
    dlib::full_object_detection sps;

    clock_weight = clock();
    clock_time = clock();

    sps = pose_model(cimg, faces[0]);

    if(sps.num_parts()>=68){
        //找到人脸
        ShapeProcessingClass shape_processing(sps);  //模型处理实例化
        state = 0;
        period_weight = clock() - clock_weight;
        clock_weight = clock();
        score += (0.5-shape_processing.eye_value)*period_weight/1000;
        score = std::max(0.0,score);
        score = std::min(SCORE_MAX, score);

        //Sub 状态判定
        if(score > SCORE_TOP){
            state = 3;
        }
        else{
            state = 4;
        }

        if(shape_processing.eye_value < RATE_BOTTOM){
            if((clock()-clock_time)/1000.0 > PERIOD_AVERAGE){
                state = 2;
            }
            if((clock()-clock_time)/1000.0 > PERIOD_SERIOUS){
                state = 1;
            }
        }
        else{
            clock_time = clock();
        }

        //Sub 末端处理(人脸切割)
        rfleft += shape_processing.shape_differ_left - MARGIN_LEFT;
        rftop += shape_processing.shape_differ_top - MARGIN_TOP;
        area_width = shape_processing.shape_width + MARGIN_LEFT + MARGIN_RIGHT;
        area_height = shape_processing.shape_height + MARGIN_TOP + MARGIN_DOWN;
    }
    else{
        std::cout << "No faces found." << std::endl;
        rfleft = 0;
        rftop = 0;
        area_width = WIN_WIDTH;
        area_height = WIN_HEIGHT;
    }
}


void DetectionProcession::lanch_handle() {
    pthread_t pth;
    pthread_create(&pth,NULL,handle_pth,NULL);
}

void *DetectionProcession::handle_pth(void *) {
    handle();
    return nullptr;
}
