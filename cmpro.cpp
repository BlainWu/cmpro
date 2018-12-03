#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>
#include <ctime>
//#include "soundpool.cpp"
//#include <SFML/Audio.hpp>

double m_ju(dlib::full_object_detection& shp);

double rk_ju(dlib::full_object_detection& shp);

double lk_ju(dlib::full_object_detection& shp);

double nod_detect(dlib::full_object_detection& shp);

std::vector<double> angle_detect(dlib::full_object_detection& shape);


double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };
std::vector<cv::Point3d> Model3D={{-7.308957,0.913869,0.000000}, {-6.775290,-0.730814,-0.012799}, {-5.665918,-3.286078,1.022951}, {-5.011779,-4.876396,1.047961}, {-4.056931,-5.947019,1.636229}, {-1.833492,-7.056977,4.061275}, {0.000000,-7.415691,4.070434}, {1.833492,-7.056977,4.061275}, {4.056931,-5.947019,1.636229}, {5.011779,-4.876396,1.047961}, {5.665918,-3.286078,1.022951}, {6.775290,-0.730814,-0.012799}, {7.308957,0.913869,0.000000}, {5.311432,5.485328,3.987654}, {4.461908,6.189018,5.594410}, {3.550622,6.185143,5.712299}, {2.542231,5.862829,4.687939}, {1.789930,5.393625,4.413414}, {2.693583,5.018237,5.072837}, {3.530191,4.981603,4.937805}, {4.490323,5.186498,4.694397}, {-5.311432,5.485328,3.987654}, {-4.461908,6.189018,5.594410}, {-3.550622,6.185143,5.712299}, {-2.542231,5.862829,4.687939}, {-1.789930,5.393625,4.413414}, {-2.693583,5.018237,5.072837}, {-3.530191,4.981603,4.937805}, {-4.490323,5.186498,4.694397}, {1.330353,7.122144,6.903745}, {2.533424,7.878085,7.451034}, {4.861131,7.878672,6.601275}, {6.137002,7.271266,5.200823}, {6.825897,6.760612,4.402142}, {-1.330353,7.122144,6.903745}, {-2.533424,7.878085,7.451034}, {-4.861131,7.878672,6.601275}, {-6.137002,7.271266,5.200823}, {-6.825897,6.760612,4.402142}, {-2.774015,-2.080775,5.048531}, {-0.509714,-1.571179,6.566167}, {0.000000,-1.646444,6.704956}, {0.509714,-1.571179,6.566167}, {2.774015,-2.080775,5.048531}, {0.589441,-2.958597,6.109526}, {0.000000,-3.116408,6.097667}, {-0.589441,-2.958597,6.109526}, {-0.981972,4.554081,6.301271}, {-0.973987,1.916389,7.654050}, {-2.005628,1.409845,6.165652}, {-1.930245,0.424351,5.914376}, {-0.746313,0.348381,6.263227}, {0.000000,0.000000,6.763430}, {0.746313,0.348381,6.263227}, {1.930245,0.424351,5.914376}, {2.005628,1.409845,6.165652}, {0.973987,1.916389,7.654050}, {0.981972,4.554081,6.301271}};
std::vector<cv::Point3d> object_pts={Model3D[33],Model3D[29],Model3D[34],Model3D[38],Model3D[13],Model3D[17],
                                     Model3D[25],Model3D[21],Model3D[55],Model3D[49],Model3D[43],Model3D[39],
                                     Model3D[45],Model3D[6]};
std::vector<cv::Point3d> reprojectsrc={cv::Point3d(10.0, 10.0, 10.0),cv::Point3d(10.0, 10.0, -10.0),
                                       cv::Point3d(10.0, -10.0, -10.0),cv::Point3d(10.0, -10.0, 10.0),
                                       cv::Point3d(-10.0, 10.0, 10.0),cv::Point3d(-10.0, 10.0, -10.0),
                                       cv::Point3d(-10.0, -10.0, -10.0),cv::Point3d(-10.0, -10.0, 10.0)};


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
    static bool is_face_found;
    static double pt = 0.0;
    static double score = 0.0;

    static std::deque<double> escon;
    std::vector<double> angled;
    std::deque<double> anglept;
    double angle;

	try
	{
        std::cout << "starting." << std::endl;

		cv::VideoCapture cap(0);
		if (!cap.isOpened()) {
			std::cerr << "Unable to connect to camera" << std::endl;
			return 1;
		}

		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
		dlib::shape_predictor pose_model;
		dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;




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
            dlib::cv_image<dlib::bgr_pixel> cimg(temp);
            std::vector<dlib::rectangle> faces = detector(cimg);
            dlib::full_object_detection sps;


            if(!faces.empty()){
                //人脸识别成功
                is_face_found=true;
                sps=pose_model(cimg, faces[0]);
                if(sps.num_parts() < 68){
                    std::cerr << "Wrong Matching" << std::endl;
                    return 13;
                }

                //检测出人脸及特征点
                double k1 = rk_ju(sps), k2 = lk_ju(sps), k,kp;
                double m3 = m_ju(sps);
                angled.clear();
                angled=angle_detect(sps);

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

                if(anglept.size()<6){
                    anglept.push_back(angled[0]);
                }
                else{
                    anglept.push_back(angled[0]);
                    anglept.pop_front();
                }
                for(auto jd:anglept){
                    angle+=jd;
                }
                angle /= (double)(anglept.size());



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
                is_face_found=false;
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
                if(is_nod) putText(showimg,"Nodding head!!!",cv::Point(50,100),CV_FONT_NORMAL,1,cvScalar(178,35,35),1,1);
                else  putText(showimg,"Normal",cv::Point(50,100),CV_FONT_NORMAL,1,cvScalar(255,255,255),1,1);
                if(is_yawn) putText(showimg,"Yawning!!!",cv::Point(50,150),CV_FONT_NORMAL,1,cvScalar(178,35,35),1,1);
                else  putText(showimg,"Normal",cv::Point(50,150),CV_FONT_NORMAL,1,cvScalar(255,255,255),1,1);
                putText(showimg, shownum[stu], cv::Point(50,200), CV_FONT_NORMAL, 1, cvScalar(255,255,255),1,1);
                if(is_face_found){
                    std::ostringstream output;
                    output << "X: " << std::to_string(angle);
                    cv::putText(showimg, output.str(), cv::Point(50, 250), CV_FONT_NORMAL, 1, cvScalar(255,0,255),1,1);
                    output.str("");
                    output << "Y: " << std::to_string(angled[1]);
                    cv::putText(showimg, output.str(), cv::Point(50, 280), CV_FONT_NORMAL, 1, cvScalar(255,0,255),1,1);
                    output.str("");
                    output << "Z: " << std::to_string(angled[2]);
                    cv::putText(showimg, output.str(), cv::Point(50, 310), CV_FONT_NORMAL, 1, cvScalar(255,0,255),1,1);
                }
                imshow("cap", showimg);
            }
        }
        doutput.close();
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


double m_ju(dlib::full_object_detection &shp) {
    long wit1=shp.part(48).x()-shp.part(54).x();
    long wit2=shp.part(48).y()-shp.part(54).y();
    double wit=sqrt(wit1*wit1+wit2*wit2);
    long hei1=shp.part(51).x()-shp.part(57).x();
    long hei2=shp.part(51).y()-shp.part(57).y();
    double hei=sqrt(hei1*hei1+hei2*hei2);
    return hei/wit;
}

double rk_ju(dlib::full_object_detection &shp) {
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

double lk_ju(dlib::full_object_detection &shp) {
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

double nod_detect(dlib::full_object_detection &shp) {
    long wit = shp.part(0).x()-shp.part(16).x();
    long hei = shp.part(24).y()-shp.part(8).y();
    long result = wit/hei;
    return result;

}

std::vector<double> angle_detect(dlib::full_object_detection &shape) {

    //result
    cv::Mat rotation_vec;                           //3 x 1
    cv::Mat rotation_mat;                           //3 x 3 R
    cv::Mat translation_vec;                        //3 x 1 T
    cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
    cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);
    //temp buf for decomposeProjectionMatrix()
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);
    cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
    cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

    std::vector<cv::Point2d> reprojectdst;
    reprojectdst.resize(8);

    std::vector<cv::Point2d> image_pts;
    image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); //#17 left brow left corner
    image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); //#21 left brow right corner
    image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); //#22 right brow left corner
    image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); //#26 right brow right corner
    image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); //#36 left eye left corner
    image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); //#39 left eye right corner
    image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); //#42 right eye left corner
    image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); //#45 right eye right corner
    image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); //#31 nose left corner
    image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); //#35 nose right corner
    image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); //#48 mouth left corner
    image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); //#54 mouth right corner
    image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); //#57 mouth central bottom corner
    image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   //#8 chin corner
    //calculate
    cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);
    //reproject
    cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);
    cv::Rodrigues(rotation_vec, rotation_mat);
    cv::hconcat(rotation_mat, translation_vec, pose_mat);
    cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

    return {euler_angle.at<double>(0),euler_angle.at<double>(1),euler_angle.at<double>(2)};
}
