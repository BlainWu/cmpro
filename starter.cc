
#ifndef CMPRO_MAIN_EXE
#define CMPRO_MAIN_EXE


#include <thread>

#include "detection_procession.h"
int main(){
    std::string textinput;
    DetectionProcession dpmain;
    //std::thread thread_loop{&DetectionProcession::loop_processing,&dpmain};
    dpmain.loop_processing();
    return 0;
}

#endif