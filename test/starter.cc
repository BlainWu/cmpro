
#ifndef CMPRO_MAIN_EXE
#define CMPRO_MAIN_EXE


#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <pthread.h>


#include "shape_processing.h"
#include "showing_window.h"



//==============================================================================
//定义部分
//Defination Part

dlib::frontal_face_detector detector;
dlib::shape_predictor pose_model;
cv::Mat showing_image;

//if value calculation loop continue
bool is_loop_continue ;
bool is_thread_on;
//state 疲劳状态:0未识别，1重度，2中度，3轻度，4正常，5警告广播 ..
//state 0~Not Leveled; 1~Degree-3; 2~Degree-2; 3~Degree-1; 4~Degree-0(Normal); 5~Alarming Signal ..
int state;

//position and size of the selected area
long rfleft, rftop, area_width, area_height;

//
double score;

unsigned long period_weight;

clock_t clock_weight ;
clock_t clock_time ;


//-----------------------------------------------------------------
double RATE_TOP;
double RATE_BOTTOM;
double ANGLE_JUDGEMENT;
double MOUTH_TOP;

double PERIOD_AVERAGE;
double PERIOD_SERIOUS;

double SCORE_TOP;
double SCORE_MAX;

//框定人脸区域上下左右预留出来的误差范围
//enlarged length in face selecting process
int MARGIN_LEFT ;
int MARGIN_RIGHT ;
int MARGIN_TOP ;
int MARGIN_DOWN ;

//识别的画面初次压缩的最终大小
//注意，和摄像头调成同比例的，建议在200以上，350以下
//first compressed rate(size)
int WIN_WIDTH ;
int WIN_HEIGHT ;

bool IS_DEBUG_WIN_ON ;      //是否显示Debug窗体
bool IS_DEBUG_WIN_DRAWED ;  //是否在Debug窗体上显示特征点（需要Debug窗体显示）
bool IS_SHOWN_WIN_ON ;       //是否显示展示窗体

std::string stin;

//各种状态的显示消息
const std::vector<std::string> ctmsg = {"ERROR","3 重度","2 中度","1 轻度","0 正常","状态异常"};
const std::vector<std::string> show_msg = {" No-Matching","degree-3","degree-2 ","degree-1","degree-0","Alarming"};



void loop_initialize(){

    //==============================================================================
    //初始化部分



    const int pin_num = 6;
    const int din_num = 8;

    std::ifstream fin;

    fin.open("../process_configure.dat");
    std::vector<int> pin;
    pin.resize(pin_num);
    for(int i=0;i<pin_num;++i){
        fin >> pin[i];
    }
    fin >> stin;
    WIN_WIDTH = pin[0];
    WIN_HEIGHT = pin[1];
    MARGIN_LEFT = pin[2];
    MARGIN_TOP = pin[3];
    MARGIN_RIGHT = pin[4];
    MARGIN_DOWN = pin[5];
    fin.close();
    fin.open("../rate_configure.dat");
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
    is_thread_on = false;

    rfleft = 0;
    rftop = 0;
    area_width = WIN_WIDTH;
    area_height = WIN_HEIGHT;
}

//thread of value calculation
void handle(){
    cv:: Mat processing_image;
    processing_image = showing_image.clone();
    cv::resize(processing_image, processing_image, cv::Size(WIN_WIDTH,WIN_HEIGHT), 0, 0, cv::INTER_LINEAR); //resize image to suitable size
    processing_image=processing_image(cv::Rect(std::max(0L,rfleft), std::max(0L,rftop),
                                               std::min(WIN_WIDTH-rfleft, area_width), std::min(WIN_HEIGHT-rftop, area_height)));
    dlib::cv_image<dlib::bgr_pixel> cimg(processing_image);
    std::vector<dlib::rectangle> faces = detector(cimg);
    dlib::full_object_detection sps;

    clock_weight = clock();
    clock_time = clock();

    sps = pose_model(cimg, faces[0]);

    if(sps.num_parts()>=68){
        //if faces points found
        ShapeProcessingClass shape_processing(sps);  //模型处理实例化
        state = 0;
        period_weight = clock() - clock_weight;
        clock_weight = clock();
        score += (0.5-shape_processing.eye_value)*period_weight/1000;
        score = std::max(0.0,score);
        score = std::min(SCORE_MAX, score);

        //Sub 状态判定
        //fatigue degree judgement
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
        //final preperation (face selection)
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
    is_thread_on = f8alse;
}

void loop_process(){
    try{
        std::cout << "starting." << std::endl;

        detector = dlib::get_frontal_face_detector();
        //model load (time consuming)
        dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;
        //cap initialization
        cv::VideoCapture cap;
        if(stin == "0"){
            cap.open(0);
        }
        else{
            cap.open("../videoraw/" + stin + ".mp4");
        }

        if (!cap.isOpened()) {
            std::cerr << "Unable to connect to camera" << std::endl;
            return;
        }
        //ShowingWindow showing_window(&cap);

        std::cout << "initialized." << std::endl;


        while(cv::waitKey(10)!='q' && is_loop_continue){

            cap >> showing_image;

            if(!is_thread_on){
                is_thread_on = true;
                std::thread thhandle(handle);
                thhandle.detach();
            }


            putText(showing_image,show_msg[state],cv::Point(10,60), cv::QT_FONT_NORMAL, 1, cvScalar(0,0,255),1,1);
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


int main(){
    loop_initialize();
    loop_process();
    return 0;
}

#endif