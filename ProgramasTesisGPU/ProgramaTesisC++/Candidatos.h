/*
 * Candidatos.h
 *
 *  Created on: Feb 6, 2017
 *      Author: luis
 */

#ifndef CANDIDATOS_H_
#define CANDIDATOS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include "opencv2/xfeatures2d.hpp"
#include<algorithm>
#include<iostream>
#include<vector>
#include "HeadDetector.h"
#include "RegionGrowing.h"
using namespace std;
using namespace cv;

class Candidatos {

	public:
		Candidatos();
		Candidatos(cv::Mat& img, Size dim) ;
		vector< cv::Mat > getCandidatosHeadDetector();
		vector< cv::Mat > getCandidatosRegionGrowing();
		vector< cv::Mat > getCandidatosTodos();

	private:
		HeadDetector met1;
		RegionGrowing met2;
		vector< cv::Mat > candidatos;
};

#endif /* CANDIDATOS_H_ */
