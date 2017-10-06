/*
 * HeadDetector.h
 *
 *  Created on: Apr 18, 2017
 *      Author: luis
 */

#ifndef HEADDETECTOR_H_
#define HEADDETECTOR_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
class HeadDetector {

public:
	HeadDetector();
	HeadDetector(cv::Mat& img);
	void detectarCabezas();
	vector< cv::Mat > getCandidatos();
	void cambiarFormato();

private:
	cv::Mat img;
	vector< cv::Mat > candidatos;
};

#endif /* HEADDETECTOR_H_ */
