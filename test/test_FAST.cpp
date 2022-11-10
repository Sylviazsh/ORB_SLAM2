#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <thread>
#include <memory>
#include <iostream>

using namespace std;
using namespace cv;

int main(){
    cv::Mat img = imread("/home/zhoush/Documents/ORB_SLAM2/test/img.png");
    vector<cv::KeyPoint> keyPoint;

    FAST(img, keyPoint, 50);
    for(auto point:keyPoint){
        circle(img, point.pt, 3, 255, -1);
    }

    imshow("FAST", img);
    waitKey(0);
}