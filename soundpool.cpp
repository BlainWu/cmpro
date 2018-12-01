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
#include <vector>
#include <queue>
#include <thread>

const int type_crt=7;
enum {
    SD_INITIANIZED=0,
    SD_DEGREE_3=1,
    SD_DEGREE_2=2,
    SD_DEGREE_1=3,
    SD_NORMAL=4,
    SD_NOD=5,
    SD_YAW=6
};


class sdplayer{
public:
    void reload(){
        bf.clear();
        //load sound file here
        //buffer.loadFromFile("sound.wav")
    }
    sdplayer(){
        reload();
        //queue.clear();
    }
    void play(int index){
        std::thread task_play(play_handle,index);
        task_play.detach();
    }
private:
    static std::vector<sf::SoundBuffer> bf;
    //queue<int> playqueue;
    static void play_handle(int index){
        sf::Sound sd;
        if(index>=0 && index<type_crt){
            sd.setBuffer(bf[index]);
            sd.play();
        }
    }

};