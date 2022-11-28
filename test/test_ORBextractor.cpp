#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <thread>
#include <memory>
#include <iostream>
#include <chrono>

#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Converter.h"

using namespace cv;
using namespace std;

void output_IC_Angle(cv::Mat M1,const vector<int> & u_max){
    std::cout << "M1 = " << std::endl
              << M1 << std::endl;
    float orientation = ORB_SLAM2::IC_Angle(M1, cv::Point2f(1,1), u_max);
    cout << "orientation = " << orientation << endl
         << endl;
}

void test_IC_Angle(const vector<int> & u_max){
    int N = 200;
    cout << "----------test_IC_Angle----------" << endl;
    cv::Mat M1 = (cv::Mat_<uchar>(3, 3) << N, 0, 0, 0, 0, 0, 0, 0, 0);
    output_IC_Angle(M1,u_max);
    M1 = (cv::Mat_<uchar>(3, 3) << 0, N, 0, 0, 0, 0, 0, 0, 0);
    output_IC_Angle(M1,u_max);
    M1 = (cv::Mat_<uchar>(3, 3) << 0, 0, N, 0, 0, 0, 0, 0, 0);
    output_IC_Angle(M1,u_max);
    M1 = (cv::Mat_<uchar>(3, 3) << 0, 0, 0, 0, 0, N, 0, 0, 0);
    output_IC_Angle(M1,u_max);
    M1 = (cv::Mat_<uchar>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, N);
    output_IC_Angle(M1,u_max);
    M1 = (cv::Mat_<uchar>(3, 3) << 0, 0, 0, 0, 0, 0, 0, N, 0);
    output_IC_Angle(M1,u_max);
    M1 = (cv::Mat_<uchar>(3, 3) << 0, 0, 0, 0, 0, 0, N, 0, 0);
    output_IC_Angle(M1,u_max);
    M1 = (cv::Mat_<uchar>(3, 3) << 0, 0, 0, N, 0, 0, 0, 0, 0);
    output_IC_Angle(M1,u_max);
    cout << endl;
}

int main()
{
    // new ORBextractor
    std::unique_ptr<ORB_SLAM2::ORBextractor> OrbExtractor = std::make_unique<ORB_SLAM2::ORBextractor>(1000, 1.2, 8, 20, 8);

    // 输入图片
    cv::Mat img = imread("/home/zhoush/Documents/ORB_SLAM2/test/img.png");
    // imshow("img", img);

    // 计算图像金字塔
    OrbExtractor->ComputePyramid(img);

    // 计算关键点八叉树
    vector<vector<KeyPoint>> allKeypoints;
    OrbExtractor->ComputeKeyPointsOctTree(allKeypoints);
    cout << "----------ComputeKeyPointsOctTree----------" << endl;
    for (uint level = 0; level < allKeypoints.size(); ++level)
    {
        cout << "level " << level << ": " << allKeypoints[level].size() << " Keypoints" << endl;
    }
    cout << endl;

    // 把特征点画到金字塔各层级图上
    for (uint level = 0; level < OrbExtractor->mvImagePyramid.size(); ++level)
    {
        for (auto point : allKeypoints[level])
            cv::circle(OrbExtractor->mvImagePyramid[level], point.pt, 3, 255, -1);
        // imshow(to_string(level), OrbExtractor->mvImagePyramid[level]);
    }

    // 测试计算灰度质心角
    //! 先把ORBextractor.cc中的PATCH_SIZE改为3，HALF_PATCH_SIZE改为1，再进行测试
    // 结果得向右为0度，向下为90度，向左为180度，向上为270度。斜对角的数字都在计算圆之外
    test_IC_Angle(OrbExtractor->umax);

    // 计算描述子
    std::vector<cv::Point> pattern;
    Mat descriptor1,descriptor2;
    const Point* pattern0 = (const Point*)ORB_SLAM2::bit_pattern_31_;
    std::copy(pattern0, pattern0 + 512, std::back_inserter(pattern));
    ORB_SLAM2::computeDescriptors(img, allKeypoints[0], descriptor1, pattern);
    ORB_SLAM2::computeDescriptors(img, allKeypoints[1], descriptor2, pattern);
    // cout << "descriptor1" << descriptor1 << endl;

    // 计算描述子距离
    //! 先取消注释ORBmatcher::DescriptorDistance中的所有PrintBinary函数，才会打印计算过程
    std::unique_ptr<ORB_SLAM2::ORBmatcher> OrbMatcher = std::make_unique<ORB_SLAM2::ORBmatcher>();
    cout << "----------DescriptorDistance----------" << endl;
    int distance = OrbMatcher->DescriptorDistance(descriptor1, descriptor2);
    cout << "distance=" << distance << endl << endl;

    // 计算BoW
    DBoW2::BowVector bow; // std::map<WordId, WordValue>
    DBoW2::FeatureVector featrue; // std::map<NodeId, std::vector<unsigned int>>
    ORB_SLAM2::ORBVocabulary *mpVocabulary = new ORB_SLAM2::ORBVocabulary();
    cout << "loadFromTextFile..." << endl;
    mpVocabulary->loadFromTextFile("/home/zhoush/Documents/ORB_SLAM2/Vocabulary/ORBvoc.txt");
    vector<cv::Mat> vCurrentDesc = ORB_SLAM2::Converter::toDescriptorVector(descriptor1);
    mpVocabulary->transform(vCurrentDesc,bow,featrue,4);
    cout << "----------ComputeBoW----------" << endl;
    // cout << "BoW=" << bow << endl << endl;
    // cout << "Featrue=" << featrue << endl;

    cv::waitKey(0);
}