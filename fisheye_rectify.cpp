#include <stdio.h>
#include <iomanip>
#include <sys/stat.h>
#include <dirent.h> //read the file below one path
//#include <unistd.h>  //access
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

#define TEST
using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {

    if(argc != 4){
        cout<<"./command path.txt out_path/ fisheye_stereo_camera.txt"<<endl;
        return -1;
    }

    string camera_param_file(argv[1]);
    string leftimg_filename = argv[2];
    string rightimg_filename = argv[3];

    FileStorage fs(camera_param_file, FileStorage::READ);
    cout<<"camera_param_file: "<<camera_param_file<<endl;
    Mat R1, R2, P1, P2, Q;
    Mat K1, K2, D1, D2, R;
    Mat T; //Vec3d T;
    Size image_size;
    Rect validRoi[2];
    Rect roi;

    cv::FileNode fn = fs.root();
    for (cv::FileNodeIterator fit = fn.begin(); fit != fn.end(); ++fit) {
      cv::FileNode item = *fit;
      std::string key = item.name();
      if (key.compare("K1") == 0) {
        fs["K1"] >> K1;
      } else if (key.compare("M1") == 0) {
        fs["M1"] >> K1; // camera matrix sometimes called K, sometimes M
      } else if (key.compare("K2") == 0) {
        fs["K2"] >> K2;
      } else if (key.compare("M2") == 0) {
        fs["M2"] >> K2;
      } else if (key.compare("D1") == 0) {
        fs["D1"] >> D1;
      } else if (key.compare("D2") == 0) {
        fs["D2"] >> D2;
      } else if (key.compare("R") == 0) {
        fs["R"] >> R;
      } else if (key.compare("T") == 0) {
        fs["T"] >> T;
      } else if (key.compare("R1") == 0) {
        fs["R1"] >> R1;
      } else if (key.compare("P1") == 0) {
        fs["P1"] >> P1;
      } else if (key.compare("R2") == 0) {
        fs["R2"] >> R2;
      } else if (key.compare("P2") == 0) {
        fs["P2"] >> P2;
      }
    }
    fs.release();
    cout<<"Finish Loading the config file."<<endl;
#ifdef TEST
    cout << "K1: " << K1 << endl;
    cout << "D1: " << D1 << endl;
    cout << "K2: " << K2 << endl;
    cout << "D2: " << D2 << endl;
    cout << "R: "  << R  << endl;
    cout << "T: "  << T  << endl;
    cout << "R1: " << R1 << endl;
    cout << "P1: " << P1 << endl;
    cout << "R2: " << R2 << endl;
    cout << "P2: " << P2 << endl;
#endif
    Mat img1 = imread(leftimg_filename, cv::IMREAD_COLOR);
    if(!img1.data){
        cout<<"Failed to load the image: "<<leftimg_filename[0]<<endl;
        return -1;
    }
    image_size = img1.size();

    if (!R1.empty() && !R2.empty() && !P1.empty() && !P2.empty()) {
        // already rectified
        std::cout << "already rectified" << endl;
    } else {
        std::cout << "rectifying..." << endl;

        cv::Size new_image_size = image_size;
        int flags = -1; // cv::CALIB_ZERO_DISPARITY;
        double balance = 0.0; // default 0.0  // f = balance * fmin + (1.0 - balance) * fmax;
        double fov_scale = 0.2; // default 1.0 // if fov_scale > 0 then f *= 1/fov_scale is applied internally
        fisheye::stereoRectify(K1, D1, K2, D2, image_size, R, T, R1, R2,
                P1, P2, Q, flags, new_image_size, balance, fov_scale);
    }

    // save
    Mat camera_intrinsic = P1.clone();
    camera_intrinsic.at<double>(0,2) = P1.at<double>(0,2);
    camera_intrinsic.at<double>(1,2) = P1.at<double>(1,2);
    cout<<"output camera instrinsic: "<<camera_intrinsic<<endl;
    FileStorage fs_out("rectified_camera_intrinsic.yml", FileStorage::WRITE);
    fs_out<<"K"<<camera_intrinsic;
    fs_out.release();

    Mat img2 = imread(rightimg_filename, cv::IMREAD_COLOR);

    if(!img2.data){
        cout<<"Failed to load the image: "<<leftimg_filename<<","<<string(rightimg_filename)<<endl;
        return -1;
    }

    cv::Mat lmapx, lmapy, rmapx, rmapy;
    cv::Mat imgU1, imgU2;
    Mat undisImgL, undisImgR;

    cv::fisheye::initUndistortRectifyMap(K1, D1, R1, P1, img1.size(), CV_32F, lmapx, lmapy);
    cv::fisheye::initUndistortRectifyMap(K2, D2, R2, P2, img2.size(), CV_32F, rmapx, rmapy);
    cv::remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
    cv::remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);

    bool is_vertial = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    Mat canvas, canvas_ori;
    double sf;
    int w, h;
    if( !is_vertial ){
        sf = 1;
        w = image_size.width;
        h = image_size.height;
        canvas.create(h, w*2, CV_8UC3);
        canvas_ori.create(h, w*2, CV_8UC3);
    } else{
        sf = 600./MAX(roi.width, roi.height);
        w = cvRound(roi.width*sf);
        h = cvRound(roi.height*sf);
        canvas.create(h*2, w, CV_8UC3);
        canvas_ori.create(h*2, w, CV_8UC3);
    }

    cv::remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
    Mat canvas_part = !is_vertial ? canvas(Rect(0, 0, w, h)) : canvas(Rect(0, 0, w, h));
    Mat canvas_part_ori = !is_vertial ? canvas_ori(Rect(0, 0, w, h)) : canvas_ori(Rect(0, 0, w, h));
    resize(imgU1, canvas_part, canvas_part.size(), 0, 0, INTER_AREA);
    resize(img1, canvas_part_ori, canvas_part_ori.size(), 0, 0, INTER_AREA);


    cv::remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);
    canvas_part = !is_vertial ? canvas(Rect(w, 0, w, h)) : canvas(Rect(0, h, w, h));
    canvas_part_ori = !is_vertial ? canvas_ori(Rect(w, 0, w, h)) : canvas_ori(Rect(0, h, w, h));
    resize(imgU2, canvas_part, canvas_part.size(), 0, 0, INTER_AREA);
    resize(img2, canvas_part_ori, canvas_part_ori.size(), 0, 0, INTER_AREA);

    if( !is_vertial )
        for( int j = 0; j < canvas.rows; j += 16 )
            line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
    else
        for( int j = 0; j < canvas.cols; j += 16 )
            line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

    if( !is_vertial )
        for( int j = 0; j < canvas_ori.rows; j += 16 )
            line(canvas_ori, Point(0, j), Point(canvas_ori.cols, j), Scalar(0, 255, 0), 1, 8);
    else
        for( int j = 0; j < canvas_ori.cols; j += 16 )
            line(canvas_ori, Point(j, 0), Point(j, canvas_ori.rows), Scalar(0, 255, 0), 1, 8);

    imshow("rectified_left_and_right", canvas);
    imshow("ori_left_and_right", canvas_ori);
    cv::imwrite("left.png",imgU1);
    cv::imwrite("right.png",imgU2);
    imwrite("rectified_left_and_right.png", canvas);

    char ch = (char)waitKey();
    if( ch == 27 || ch == 'q' || ch == 'Q' )
        return 0;
    return 0;
}
