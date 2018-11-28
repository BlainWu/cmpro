#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <ctime>

double rk_ju(dlib::full_object_detection& shp){
    if(shp.num_parts()<68){
        std::cout << "face model missing" << std::endl;
        return -1.0;
    }
    long le1=shp.part(39).x()-shp.part(36).x();
    long le2=shp.part(39).y()-shp.part(36).y();
    double le=sqrt(le1*le1+le2*le2);
    long da1=shp.part(41).x()-shp.part(37).x();
    long da2=shp.part(41).y()-shp.part(37).y();
    long db1=shp.part(40).x()-shp.part(38).x();
    long db2=shp.part(40).y()-shp.part(38).y();
    double da=sqrt(da1*da1+da2*da2);
    double db=sqrt(db1*db1+db2*db2);
    double k;
    if(le<1){
        std::cout << "eyes detect mistake" << std::endl;
        return -1.0;
    }
    k=(da+db)/le;
    return k;
}

double lk_ju(dlib::full_object_detection& shp){
    if(shp.num_parts()<68){
        std::cout << "face model missing" << std::endl;
        return -1.0;
    }

    long le1=shp.part(45).x()-shp.part(42).x();
    long le2=shp.part(45).y()-shp.part(42).y();
    double le=sqrt(le1*le1+le2*le2);
    long da1=shp.part(47).x()-shp.part(43).x();
    long da2=shp.part(47).y()-shp.part(43).y();
    long db1=shp.part(46).x()-shp.part(44).x();
    long db2=shp.part(46).y()-shp.part(44).y();
    double da=sqrt(da1*da1+da2*da2);
    double db=sqrt(db1*db1+db2*db2);
    double k;

    if(le<1){
        std::cout << "eyes detect mistake" << std::endl;
        return -1.0;
    }
    k=(da+db)/le;
    return k;
}

int main(){

    std::ifstream fin;
    fin.open("../conf.dat");
    std::vector<int> pin;
    pin.resize(3);
    for(int i=0;i<2;++i){
        fin >> pin[i];
    }
    fin.close();


    const int WIN_WIDTH=pin[0] ;
    const int WIN_HEIGHT=pin[1] ;

    const long MG_LEFT=10L ;
    const long MG_RIGHT=10L ;
    const long MG_TOP=8L ;
    const long MG_DOWN=8L ;

    const std::vector<std::string> stepmsg={"请确认正常状态……，按q确认","请确认疲惫状态……，按q确认","请确认闭眼状态……，按q确认"};
    const std::vector<std::string> showmsg={"Please check Degree-0 condition.",
                                            "Please check \'Tired\' condition.",
                                            "Please check eyes-closed condition."};
    const std::string showmsg2="Press Q to confirm";
    static long fleft = 0, fright = WIN_WIDTH, ftop = 0, fdown = WIN_HEIGHT;
    static long rfleft = 0, rftop = 0, dleft = 0, dtop = 0;
    try
    {
        std::cout << "starting." << std::endl;
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Unable to connect to camera" << std::endl;
            return 1;
        }
        std::vector<double> st;
        st.resize(3);
        cv::Mat showimg;

        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor pose_model;
        dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;

        for(int step=0;step<3;++step){
            std::cout << stepmsg[step] << std::endl;

            double k1=-1,k2=-1;
            while(cv::waitKey(30) != 'q'){
                cv::Mat temp;
                cap >> showimg;
                temp=showimg.clone();
                cv::resize(temp, temp, cv::Size(WIN_WIDTH,WIN_HEIGHT), 0, 0, CV_INTER_LINEAR);
                temp=temp(cv::Rect((int)std::max(0L,rfleft), (int)std::max(0L,rftop), (int)std::min((long)WIN_WIDTH-rfleft,
                                                                                                    fright-fleft), (int)std::min((long)WIN_HEIGHT-rftop, fdown-ftop)));
                dlib::cv_image<dlib::bgr_pixel> cimg(temp);
                std::vector<dlib::rectangle> faces = detector(cimg);
                dlib::full_object_detection sps;

                putText(showimg, showmsg[step], cv::Point(10,60), CV_FONT_NORMAL, 1, cvScalar(0,0,255),1,1);
                putText(showimg, showmsg2, cv::Point(10,110), CV_FONT_NORMAL, 1, cvScalar(0,0,255),1,1);

                if(!faces.empty()) {
                    //人脸识别成功
                    sps = pose_model(cimg, faces[0]);
                    if (sps.num_parts() < 68) {
                        std::cerr << "Wrong Matching" << std::endl;
                        return 13;
                    }

                    k1 = rk_ju(sps), k2 = lk_ju(sps);
                    fleft=sps.part(0).x();
                    fright=sps.part(16).x();
                    ftop=std::min(sps.part(19).y(),sps.part(24).y());
                    fdown=sps.part(9).y();
                    for(unsigned long i=1;i<27;i++){
                        fleft=std::min(fleft,sps.part(i).x());
                        fright=std::max(fright,sps.part(i).x());
                        ftop=std::min(ftop,sps.part(i).y());
                        fdown=std::max(fdown,sps.part(i).y());
                    }

                    dleft=fleft-MG_LEFT;
                    dtop=ftop-MG_TOP;

                    fleft=std::max(0L,fleft-MG_LEFT);
                    fright=std::min((long)WIN_WIDTH,fright+MG_RIGHT);
                    ftop=std::max(0L,ftop-MG_TOP);
                    fdown=std::min((long)WIN_HEIGHT,fdown+MG_DOWN);

                    rfleft+=dleft;
                    rftop+=dtop;

                    int a=(int)rfleft, b=(int)rftop, c=(int)(rfleft+fright-fleft), d=(int)(rftop+fdown-ftop);
                    a *= showimg.cols/WIN_HEIGHT;
                    b *= showimg.rows/WIN_HEIGHT;
                    c *= showimg.cols/WIN_WIDTH;
                    d *= showimg.rows/WIN_HEIGHT;

                    cv::rectangle(showimg,
                                  cvPoint(a+45,b+45),
                                  cvPoint(c+45,d+35),
                                  cvScalar(255, 0, 0));
                }
                else{
                    //人脸识别失败
                    fleft=0;
                    fright=WIN_WIDTH;
                    ftop=0;
                    fdown=WIN_HEIGHT;
                    rfleft=0;
                    rftop=0;
                }
                imshow("cap",showimg);
            }
            if(k1<0 || k2<0){
                step--;
                continue;
            }
            else{
                st[step]=(k1+k2)/2;
            }

        }
        std::ofstream flout;
        flout.open("../aout.txt",std::ios::out);
        flout << (2*st[0]+st[1])/3 << std::endl;
        flout << (st[1]+3*st[2])/4 << std::endl;
        flout.close();
        std::cout << "采集数据结束" << std::endl;
    }
    catch (dlib::serialization_error& e)
    {
        std::cout << std::endl << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    return 0;
}