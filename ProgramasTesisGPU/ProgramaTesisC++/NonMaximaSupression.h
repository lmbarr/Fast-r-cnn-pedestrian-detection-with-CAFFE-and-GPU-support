/*
 * NonMaximaSupression.h
 *
 *  Created on: Feb 7, 2017
 *      Author: luis
 */

#ifndef NONMAXIMASUPRESSION_H_
#define NONMAXIMASUPRESSION_H_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
vector<vector<Point> > non_max_suppression_fast(Mat &boxes, float overlapThresh);
vector<vector<Point> > non_max_suppression(Mat &lista_bb, vector<float>* confidencias, float solapamiento);
float IoU(Mat & elemento1, Mat & elemento2);
float intersection(Mat & elemento1, Mat & elemento2);
float unionn(Mat & elemento1, Mat & elemento2, float intersectionArea);

class MiEstructura {

public:
	MiEstructura(float confidencia, cv::Mat& bb);
	float confidencia;
	cv::Mat bb;


    bool operator==(const MiEstructura &obj) const {
    	float EPSILON = 0.00001;
        bool valor1 = fabs(obj.confidencia - confidencia) < EPSILON;
        bool valor2 = fabs(obj.bb.at<float>(0) - bb.at<float>(0)) < EPSILON;
        bool valor3 = fabs(obj.bb.at<float>(1) - bb.at<float>(1)) < EPSILON;
        bool valor4 = fabs(obj.bb.at<float>(2) - bb.at<float>(2)) < EPSILON;
        bool valor5 = fabs(obj.bb.at<float>(3) - bb.at<float>(3)) < EPSILON;
/*        cout<<fabs(obj.bb.at<float>(0) - bb.at<float>(0))<<endl;
        cout<<valor1<<valor2<<valor3<<valor4<<valor5<<endl;
    	cout<<obj.confidencia<<confidencia<<endl;
    	cout<<obj.bb<<bb<<endl;*/
        return (valor1 && valor2 && valor3 && valor4 && valor5);
    }
};

struct MyStruct
{
    float confidencia;
    cv::Mat bb;
    MyStruct(float p, cv::Mat & box) : confidencia(p), bb(box) {}
};







#endif /* NONMAXIMASUPRESSION_H_ */
