/*
 * sdplayer.play(sound_index)
 * sound_index:
 * 0 SD_INITIALIZED
 * 1 DEGREE_3
 * 2 DEGREE_2
 * 3 DEGREE_1
 * 4 SD_NORMAL
 * 5 SD_NOD
 * 6 SD_YAW
 * */

#include <SFML/Audio.hpp>
#include <thread>

const int type_crt=6;
enum {
    SD_INITIANIZED=0,
    SD_DEGREE_3=1,
    SD_DEGREE_2=2,
    SD_DEGREE_1=3,
    SD_NOD=4,
    SD_YAW=5
};


class sdplayer{
public:
    void reload(){
        sdplayer();
    }
    sdplayer(){
        sf::SoundBuffer tmp;
        bf0.loadFromFile("../sdpool/0.wav");
        bf1.loadFromFile("../sdpool/1.wav");
        bf2.loadFromFile("../sdpool/2.wav");
        bf3.loadFromFile("../sdpool/3.wav");
        bf4.loadFromFile("../sdpool/4.wav");
        bf5.loadFromFile("../sdpool/5.wav");

    }
    void play(int index){
        std::thread task_play(play_handle,index);
        task_play.detach();
    }

private:

    static sf::SoundBuffer bf0,bf1,bf2,bf3,bf4,bf5;
    //queue<int> playqueue;
    static void play_handle(int index){
        sf::Sound sd;
        switch (index){
            case 0: sd.setBuffer(bf0);break;
            case 1: sd.setBuffer(bf1);break;
            case 2: sd.setBuffer(bf2);break;
            case 3: sd.setBuffer(bf3);break;
            case 4: sd.setBuffer(bf4);break;
            case 5: sd.setBuffer(bf5);break;
        }
        sd.play();
    }

};