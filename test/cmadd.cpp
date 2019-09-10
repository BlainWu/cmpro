#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <ctime>
#include "angledata.cpp"

double rk_ju(dlib::full_object_detection& shp);

double lk_ju(dlib::full_object_detection& shp);

std::vector<double> angle_detect(dlib::full_object_detection& shape);

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

    const std::vector<std::string> stepmsg={"请确认正常状态……，按q确认","请确认疲惫状态……，按q确认",
                                            "请确认闭眼状态……，按q确认","请确认点头状态……，按q确认"};
    const std::vector<std::string> showmsg={"Please check Degree-0 condition.",
                                            "Please check \'Tired\' condition.",
                                            "Please check eyes-closed condition.",
                                            "Please check \'nodding\' condition"};
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
        double poseangle=0;

        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor pose_model;
        dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;

        for(int step=0;step<4;++step){
            std::cout << stepmsg[step] << std::endl;
            double angled;
            std::deque<double> anglept;
            anglept.resize(3);
            std::vector<double> angle;
            angle.resize(3);
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
                    angled=angle_detect(sps)[0];
                    if(step==3){
                        if(anglept.size()<6){
                            anglept.push_back(angled);
                        }
                        else{
                            anglept.push_back(angled);
                            anglept.pop_front();
                        }
                    }
                    for(auto jd:anglept){
                        poseangle+=jd;
                    }
                    poseangle /= (double)(anglept.size());

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
                    a *= showimg.cols/WIN_WIDTH;
                    b *= showimg.rows/WIN_HEIGHT;
                    c *= showimg.cols/WIN_WIDTH;
                    d *= showimg.rows/WIN_HEIGHT;

                    cv::rectangle(showimg,
                                  cvPoint(a,b),
                                  cvPoint(c,d+showimg.rows*0.08),
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
        flout << poseangle << std::endl;
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