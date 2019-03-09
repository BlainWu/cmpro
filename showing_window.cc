//
// Created by kanae on 2/14/19.
//

#include "showing_window.h"

void ShowingWindow::init() {
    is_inputed = false;
    text_index_max=0;
    text_x_base = 50;
    text_y_base = 100;
    text_y_interval = 50;
}

ShowingWindow::ShowingWindow(cv::VideoCapture* cap) {
    init();
    showing_image = original_image;
}

void ShowingWindow::TextAdd(std::string text_content) {
    text_index_max += 1;
    inputed_texts.push_back(text_content);
}

void ShowingWindow::TextClear() {
    text_index_max = 0;
    inputed_texts.clear();
}

void ShowingWindow::soundload() {
    bf.clear();
    bf.resize(type_crt);
    sd.resize(type_crt);
    for(int i=0;i<type_crt;++i){
        std::string fm;
        fm=(std::string)("../sdpool/") + std::to_string(i) + (std::string)(".wav");
        if(!bf[i].loadFromFile(fm))
            std::cerr<< "Sound files load Error"<<std::endl;
        sd[i].setBuffer(bf[i]);
    }
}

void ShowingWindow::soundplay(int index) {
    sd[index].play();
}
