#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>
#include <ctime>
#include "soundpool.cpp"
#include <SFML/Audio.hpp>


using namespace dlib;

double m_ju(full_object_detection& shp){
    long wit1=shp.part(48).x()-shp.part(54).x();
    long wit2=shp.part(48).y()-shp.part(54).y();
    double wit=sqrt(wit1*wit1+wit2*wit2);
    long hei1=shp.part(51).x()-shp.part(57).x();
    long hei2=shp.part(51).y()-shp.part(57).y();
    double hei=sqrt(hei1*hei1+hei2*hei2);
    return hei/wit;
}

double rk_ju(full_object_detection& shp){
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

double lk_ju(full_object_detection& shp){
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

double nod_detect(full_object_detection& shp){
    long wit = shp.part(0).x()-shp.part(16).x();
    long hei = shp.part(24).y()-shp.part(8).y();
    long result = wit/hei;
    return result;

}

int main()
{
    std::ifstream fin;
    fin.open("../conf.dat");
    std::vector<int> pin;
    pin.resize(3);
    for(int i=0;i<3;++i){
        fin >> pin[i];
    }
    fin.close();
    fin.open("../aout.txt");
    std::vector<double> din;
    din.resize(2);
    for(int i=0;i<2;++i){
        fin >> din[i];
    };
    std::ofstream fout;
    fout.open("../conf.dat");
    fout << pin[0] << std::endl;
    fout << pin[1] << std::endl;
    fout << pin[2]+1 << std::endl;
    fout.close();
    //框定人脸区域上下左右预留出来的误差范围
    const long MG_LEFT=10L ;
    const long MG_RIGHT=10L ;
    const long MG_TOP=8L ;
    const long MG_DOWN=8L ;

    //识别的画面初次压缩的最终大小
    //注意，和摄像头调成同比例的，建议在200以上，350以下
    const int WIN_WIDTH=pin[0] ;
    const int WIN_HEIGHT=pin[1] ;

    //判定临界
    const double SCORE_MAX=350.0;   //score疲劳度上限
    const double JS_1=250.0 ;    //最终判定中度->中度的临界

    //眼部判定为疲劳/不疲劳的临界
    const double JM_1 = din[0];
    //眼部判定为闭合的临界
    const double JM_2 = din[1];

    //k浮动系数
    const std::vector<double> FLK={1.0,0.8};
    const unsigned long FL_EXDT=5000000;    //浮动判定延续时间
    //点头判定阀值
    const double NOD_JU = 0.46;
    //口部判定阈值
    const double TH_MJU =0.6;

    //中度/重度延时持续时间
    const std::vector<unsigned long> TM_DELAY={0,3000000,5000000};
    const std::vector<unsigned long> TM_INTV={0,2000000,3000000};

    //各种状态的显示消息
    const std::vector<std::string> ctmsg = {" No matching","3 重度","2 中度","1 轻度","0 正常","状态异常"};
    const std::vector<std::string> shownum = {" No matching","degree-3","degree-2 ","degree-1","degree-0","Alarming"};

    const bool IS_DEBUG_WIN_ON = true ;           //是否显示Debug窗体
    const bool IS_DEBUG_WIN_DRAWED = true ;      //是否在Debug窗体上显示特征点（需要Debug窗体显示）
    const bool IS_SHOWN_WIN_ON = true;            //是否显示展示窗体

    //=================================================================================================

    std::string dname = std::to_string(pin[2]);
    dname = "../recdat/" + dname + ".txt";
    std::ofstream doutput;
    doutput.open(dname,std::ios::out);

    //计时器
    //tk：用于哈欠延时
    static clock_t tk_rec = clock();
    static clock_t tk_cur = clock();
    //tc：用于疲劳判定
    clock_t tc_rec = clock();
    clock_t tc_cur ;

    //面部切割过程量
    static long fleft = 0, fright = WIN_WIDTH, ftop = 0, fdown = WIN_HEIGHT;
    static long rfleft = 0, rftop = 0, dleft = 0, dtop = 0;

    //状态参数
    static int stu = 0;   //疲劳状态编号
    static int kst = 0;   //检测浮动严格程度
    static bool is_ecsd = false; //当前帧是否处于「中度/重度疲劳」状态
    static  bool is_nod = false ;//是否在点头状态
    static bool is_yawn = false ;//是否在打哈欠
    //static bool is_eces = false; //当前帧是否处于「中度/重度疲劳延时」状态
    static double pt = 0.0;
    static double score = 0.0;
    static std::deque<double> escon;

	try
	{
        std::cout << "starting." << std::endl;

		cv::VideoCapture cap(0);
		if (!cap.isOpened()) {
			std::cerr << "Unable to connect to camera" << std::endl;
			return 1;
		}

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;

        std::cout << "initialized." << std::endl;

        // Grab and process frames until the main window is closed by the user.
        while (cv::waitKey(30) != 27)
        {
            cv::Mat temp, showimg;
            cap >> temp;
	        showimg = temp.clone();
            cv::resize(temp, temp, cv::Size(WIN_WIDTH,WIN_HEIGHT), 0, 0, CV_INTER_LINEAR); //resize image to suitable size
            
            temp=temp(cv::Rect(0, 0, WIN_WIDTH,WIN_HEIGHT));
            temp=temp(cv::Rect((int)std::max(0L,rfleft), (int)std::max(0L,rftop), (int)std::min((long)WIN_WIDTH-rfleft,
                    fright-fleft), (int)std::min((long)WIN_HEIGHT-rftop, fdown-ftop)));
            cv_image<bgr_pixel> cimg(temp);
            std::vector<rectangle> faces = detector(cimg);
            full_object_detection sps;


            if(!faces.empty()){
                //人脸识别成功
                sps=pose_model(cimg, faces[0]);
                if(sps.num_parts() < 68){
                    std::cerr << "Wrong Matching" << std::endl;
                    return 13;
                }

                //检测出人脸及特征点
                double k1 = rk_ju(sps), k2 = lk_ju(sps), k,kp;
                double m3 = m_ju(sps);

                double nod_ju = nod_detect(sps);
                //判断是否为点头
                if(nod_ju> NOD_JU ) is_nod = true ;
                else is_nod = false;


                if(kst == 1){
                    tk_cur = clock();
                    //超时
                    if((tk_cur-tk_rec)>FL_EXDT){
                        kst = 0;
                    }
                }
                //口部判定->检测浮动严格程度
                if(m3 > TH_MJU){
                    kst = 1;
                    is_yawn = true;
                    tk_rec=clock();
                }
                else is_yawn = false ;

                //眼睛开闭程度不统一，判为异常驾驶状态->检测浮动严格程度
                if(abs(k1-k2)>0.2){
                    stu = 5;
                    kst = 1;
                    tk_rec=clock();
                }

                k = (k1+k2)/2.0;
                if(escon.size()<5){
                    escon.push_back(k);
                }
                else{
                    escon.push_back(k);
                    escon.pop_front();
                }
                kp = escon[0];
                for(double rk:escon){
                    kp=std::min(kp,rk);
                }
                if(kp<JM_2 && !is_ecsd){
                    is_ecsd = true;
                    tc_rec = clock();
                }

                if(kp>JM_2 && is_ecsd){
                    stu=0;
                    is_ecsd=false;
                }

                if(is_ecsd){
                    tc_cur = clock();
                    if(tc_cur-tc_rec>TM_DELAY[2]){
                        stu=2;
                    }
                    if(tc_cur-tc_rec>TM_DELAY[3]){
                        stu=1;
                    }
                }
                if(stu==2 || stu==1){
                    std::cout << ctmsg[stu] << std::endl;
                    doutput << stu << "," << k << "," << kp <<  std::endl;
                }
                else{
                    pt = k * FLK[kst] * 100;
                    //判断算法
                    if(pt>=100*JM_1 && score<0.002){
                        score = 0.0;
                    }
                    if(pt>=100*JM_1 && score>0.001){
                        score = (30.0-pt)+score;
                    }
                    if(pt<100*JM_1 && score<SCORE_MAX+0.01){
                        score = (55-pt)+score;
                    }
                    if(pt<100*JM_1 && score>SCORE_MAX){
                        score = SCORE_MAX;
                        score = (55.0-pt)+score;
                    }

                    if(score<JS_1){
                        stu = 4;
                    }
                    else{
                        stu = 3;
                    }
                    std::cout << ctmsg[stu] << "," << m3 << "," << kst << "," << pt << "," << score << std::endl;
                    doutput << stu << "," << k << "," << score << std::endl;
                }



                //面部切割finalA(识别)
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

                //特征点Debug显示
                if(IS_DEBUG_WIN_ON && IS_DEBUG_WIN_DRAWED) {
                    for (int i = 0; i < 17; i++) {
                        cv::circle(temp, cvPoint(sps.part(i).x(), sps.part(i).y()), 2, cvScalar(0, 0, 0), -1);
                    }
                    for (int i = 17; i < 25; i++) {
                        cv::circle(temp, cvPoint(sps.part(i).x(), sps.part(i).y()), 2, cvScalar(0, 120, 40), -1);
                    }
                    for (int i = 27; i < 36; i++) {
                        cv::circle(temp, cvPoint(sps.part(i).x(), sps.part(i).y()), 2, cvScalar(0, 0, 255), -1);
                    }
                    for (int i = 36; i < 48; i++) {
                        cv::circle(temp, cvPoint(sps.part(i).x(), sps.part(i).y()), 2, cvScalar(255, 0, 0), -1);
                    }
                    for (int i = 48; i < 68; i++) {
                        cv::circle(temp, cvPoint(sps.part(i).x(), sps.part(i).y()), 2, cvScalar(0, 255, 0), -1);
                    }
                }

            }
            else{
                //人脸识别失败
                std::cout << "No faces found." << std::endl;

                //面部切割finalB(未识别)
                fleft=0;
                fright=WIN_WIDTH;
                ftop=0;
                fdown=WIN_HEIGHT;
                rfleft=0;
                rftop=0;
            }


            if(IS_DEBUG_WIN_ON){
                imshow("Debug",temp);
            }


            if(IS_SHOWN_WIN_ON){
                if(is_yawn) putText(showimg,"Yawning!!!",cv::Point(50,150),CV_FONT_NORMAL,1,cvScalar(178,35,35),1,1);
                else  putText(showimg,"Normal",cv::Point(50,150),CV_FONT_NORMAL,1,cvScalar(255,255,255),1,1);
                if(is_nod) putText(showimg,"Nodding head!!!",cv::Point(50,100),CV_FONT_NORMAL,1,cvScalar(178,35,35),1,1);
                else  putText(showimg,"Normal",cv::Point(50,100),CV_FONT_NORMAL,1,cvScalar(255,255,255),1,1);

                putText(showimg, shownum[stu], cv::Point(50,200), CV_FONT_NORMAL, 1, cvScalar(255,255,255),1,1);
                imshow("cap", showimg);
            }
        }
        doutput.close();
    }
    catch (serialization_error& e)
    {
        std::cout << std::endl << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    return 0;
}
