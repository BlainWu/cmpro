//
// Created by kanae on 2/14/19.
//

#ifndef CMPRO_SHOWING_WINDOW_H
#define CMPRO_SHOWING_WINDOW_H

#include <opencv2/opencv.hpp>
#include <SFML/Audio.hpp>
#include <string>
#include <vector>

class ShowingWindow {
public:

    ShowingWindow(cv::VideoCapture* cap);

    void TextAdd(std::string text_content);

    void TextClear();

    void soundplay(int index);

    bool is_inputed;
private:
    cv::VideoCapture showing_cap;

    cv::Mat original_image;
    cv::Mat showing_image;

    int text_x_base;
    int text_y_base;
    int text_y_interval;
    std::vector<std::string> inputed_texts;
    int text_index_max;


    std::vector<sf::SoundBuffer> bf;
    std::vector<sf::Sound> sd;
    const unsigned long type_crt=6;
    void soundload();

    bool is_showed;

    void init();

};


#endif //CMPRO_SHOWING_WINDOW_H
