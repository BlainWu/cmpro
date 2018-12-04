/*
 * sdplayer.play(sound_index)
 * sound_index:
 * 0 SD_INITIALIZED
 * 1 DEGREE_3
 * 2 DEGREE_2
 * 3 DEGREE_1
 * 4 SD_NOD
 * 5 SD_YAW
 * */

#include <SFML/Audio.hpp>
#include <iostream>
#include <thread>
#include <string>

const int type_crt=6;
std::vector<sf::SoundBuffer> bf;
std::vector<sf::Sound> sd;
void soundload(){
    bf.clear();
    bf.resize(type_crt);
    sd.resize(type_crt);
    for(int i=0;i<type_crt;++i){
        std::string fm;
        fm=(std::string)("../sdpool/") + std::to_string(i) + (std::string)(".wav");
        if(!bf[i].loadFromFile(fm))
            std::cout<< "sound files load error"<<std::endl;
        sd[i].setBuffer(bf[i]);
    }
}
void soundplay(int index){
    sd[index].play();
    std::cout << "played" <<std::endl;
}

