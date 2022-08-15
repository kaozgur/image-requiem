#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;

#define NONE 0  
#define HARD 1  
#define SOFT 2  
#define GARROT 3 




float sgn(float x)
{
    float res = 0;
    if (x == 0)
    {
        res = 0;
    }
    if (x > 0)
    {
        res = 1;
    }
    if (x < 0)
    {
        res = -1;
    }
    return res;
}

float soft_shrink(float d, float T)
{
    float res;
    if (fabs(d) > T)
    {
        res = sgn(d) * (fabs(d) - T);
    }
    else
    {
        res = 0;
    }

    return res;
}

float hard_shrink(float d, float T)
{
    float res;
    if (fabs(d) > T)
    {
        res = d;
    }
    else
    {
        res = 0;
    }

    return res;
}

float Garrot_shrink(float d, float T)
{
    float res;
    if (fabs(d) > T)
    {
        res = d - ((T * T) / d);
    }
    else
    {
        res = 0;
    }

    return res;
}

static void cvHaarWavelet(Mat& src, Mat& dst, int NIter)
{
    float c, dh, dv, dd;
    assert(src.type() == CV_32FC1);
    assert(dst.type() == CV_32FC1);
    int width = src.cols;
    int height = src.rows;
    for (int k = 0; k < NIter; k++)
    {
        for (int y = 0; y < (height >> (k + 1)); y++)
        {
            for (int x = 0; x < (width >> (k + 1)); x++)
            {
                c = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
                dst.at<float>(y, x) = c;

                dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
                dst.at<float>(y, x + (width >> (k + 1))) = dh;

                dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
                dst.at<float>(y + (height >> (k + 1)), x) = dv;

                dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
                dst.at<float>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;
            }
        }
        dst.copyTo(src);
    }
}

static void cvInvHaarWavelet(Mat& src, Mat& dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50)
{
    float c, dh, dv, dd;
    assert(src.type() == CV_32FC1);
    assert(dst.type() == CV_32FC1);
    int width = src.cols;
    int height = src.rows;

    for (int k = NIter; k > 0; k--)
    {
        for (int y = 0; y < (height >> k); y++)
        {
            for (int x = 0; x < (width >> k); x++)
            {
                c = src.at<float>(y, x);
                dh = src.at<float>(y, x + (width >> k));
                dv = src.at<float>(y + (height >> k), x);
                dd = src.at<float>(y + (height >> k), x + (width >> k));

                // (shrinkage)
                switch (SHRINKAGE_TYPE)
                {
                case HARD:
                    dh = hard_shrink(dh, SHRINKAGE_T);
                    dv = hard_shrink(dv, SHRINKAGE_T);
                    dd = hard_shrink(dd, SHRINKAGE_T);
                    break;
                case SOFT:
                    dh = soft_shrink(dh, SHRINKAGE_T);
                    dv = soft_shrink(dv, SHRINKAGE_T);
                    dd = soft_shrink(dd, SHRINKAGE_T);
                    break;
                case GARROT:
                    dh = Garrot_shrink(dh, SHRINKAGE_T);
                    dv = Garrot_shrink(dv, SHRINKAGE_T);
                    dd = Garrot_shrink(dd, SHRINKAGE_T);
                    break;
                }


                dst.at<float>(y * 2, x * 2) = 0.5 * (c + dh + dv + dd);
                dst.at<float>(y * 2, x * 2 + 1) = 0.5 * (c - dh + dv - dd);
                dst.at<float>(y * 2 + 1, x * 2) = 0.5 * (c + dh - dv - dd);
                dst.at<float>(y * 2 + 1, x * 2 + 1) = 0.5 * (c - dh - dv + dd);
            }
        }
        Mat C = src(Rect(0, 0, width >> (k - 1), height >> (k - 1)));
        Mat D = dst(Rect(0, 0, width >> (k - 1), height >> (k - 1)));
        D.copyTo(C);
    }
}

static void WaveletTransform(Mat& img)
{
    int n = 0;
    const int NIter = 1;
    Mat girdi_img = img.clone();
    Mat GrayFrame = Mat(girdi_img.rows, girdi_img.cols, CV_8UC1);
    Mat Src = Mat(girdi_img.rows, girdi_img.cols, CV_32FC1);
    Mat Dst = Mat(girdi_img.rows, girdi_img.cols, CV_32FC1);
    Mat Temp = Mat(girdi_img.rows, girdi_img.cols, CV_32FC1);
    Mat Filtered = Mat(girdi_img.rows, girdi_img.cols, CV_32FC1);

    Dst = 0;

    cvtColor(girdi_img, GrayFrame, cv::COLOR_BGRA2GRAY);
    GrayFrame.convertTo(Src, CV_32FC1);
    cvHaarWavelet(Src, Dst, NIter);

    Dst.copyTo(Temp);

    cvInvHaarWavelet(Temp, Filtered, NIter, GARROT, 30);

    //imshow("DWT1",girdi_img);


    double M = 0, m = 0;

    minMaxLoc(Dst, &m, &M);
    if ((M - m) > 0) 
    { 
        Dst = Dst * (1.0 / (M - m)) - m / (M - m); 
    }
    
   
    imshow("Coeff", Dst);
    Dst.convertTo(Dst, CV_8UC3, 255.0);
    imwrite("Resources/Coeff.jpg", Dst);

    minMaxLoc(Filtered, &m, &M);
    if ((M - m) > 0) 
    { 
        Filtered = Filtered * (1.0 / (M - m)) - m / (M - m); 
    }

    
    
    imshow("Filtered Version", Filtered);
    Filtered.convertTo(Filtered, CV_8UC3, 255.0);
    imwrite("Resources/Filtered.jpg", Filtered);


}

Mat kuantalama(Mat img, int kuantalama_renkleri) 
{

    Mat src = img.clone();  //copy the main image
    Mat data = Mat::zeros(src.cols * src.rows, 3, CV_32F);  //matrix to keep all the data oof all pixels
    Mat bestLabels, centers, clustered; //k means output
    vector<Mat> bgr;    //to keep bgr channels
    split(src, bgr);

    //for getting all the columns and raws data for the k means
    for (int i = 0; i < src.cols * src.rows; i++) {
        data.at<float>(i, 0) = bgr[0].data[i] / 255.0;
        data.at<float>(i, 1) = bgr[1].data[i] / 255.0;
        data.at<float>(i, 2) = bgr[2].data[i] / 255.0;
    }

    int K = kuantalama_renkleri;    //cluster amount
    kmeans(data, K, bestLabels,
        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0),
        3, KMEANS_PP_CENTERS, centers);

    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);

    clustered = Mat(src.rows, src.cols, CV_32F);


    Vec3f* p = data.ptr<Vec3f>();
    for (size_t i = 0; i < data.rows; i++) {
        int center_id = bestLabels.at<int>(i);
        p[i] = centers.at<Vec3f>(center_id);
    }

    clustered = data.reshape(3, src.rows);
    imshow("Clustered Version", clustered);
    clustered.convertTo(clustered, CV_8UC3, 255.0);
    imwrite("Resources/Clustered.jpg", clustered);
    return clustered;
}

Mat renk_donusumu(int width, int height, Mat img)
{
    Mat cikti_img = img;
    Vec3b donusturulecek_pixel;
    Vec3b donusturulecek_pixel2;
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            donusturulecek_pixel = cikti_img.at<Vec3b>(Point(i, j));
            donusturulecek_pixel2 = cikti_img.at<Vec3b>(Point(i, j));

            donusturulecek_pixel[0] = 0.299 * (donusturulecek_pixel2[2] - donusturulecek_pixel2[1]) + donusturulecek_pixel2[1] + 0.114 * (donusturulecek_pixel2[0] - donusturulecek_pixel2[1]);
            donusturulecek_pixel[1] = 0.564 * (donusturulecek_pixel2[0] - donusturulecek_pixel[0]);
            donusturulecek_pixel[2] = 0.713 * (donusturulecek_pixel2[2] - donusturulecek_pixel[0]);

            cikti_img.at<Vec3b>(Point(i, j)) = donusturulecek_pixel;


        }
    }
    return cikti_img;
}



int main(int ac, char** av) {

    string path = "Resources/test1.jpg";
    Mat girdi_img = imread(path);
    Mat kr,temp;
    
    Mat cikti_img = girdi_img.clone();

    if (empty(girdi_img))
    {
        cout << "Cannot found an Image\n";
        return -1;
    }
    else
    {
        cout << "Resolution: " << cikti_img.size().width << "x" << cikti_img.size().height << endl;

        int width = (int)cikti_img.size().width;
        int height = (int)cikti_img.size().height;

        //cikti_img = renk_donusumu(width, height, cikti_img);
        imwrite("Resources/output_img.jpeg", cikti_img);
        WaveletTransform(cikti_img);
        Mat filtered = imread("Resources/Filtered.jpg");
        kr = kuantalama(filtered, 8);
        
        
        
        
    }
    waitKey(0);
    return 0;
}