/*
 * RegionGrowing.h
 *
 *  Created on: Apr 16, 2017
 *      Author: luis
 */

#ifndef REGIONGROWING_H_
#define REGIONGROWING_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/videoio.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;

class RegionGrowing {

public:
	RegionGrowing();
	RegionGrowing(cv::Mat& img, float step, Size dim);
	void empezar_calculo();
	vector< cv::Mat > getCandidatos();

private:
	cv::Mat img;
	float umbral_inicial;
	float step;
	vector< cv::Mat > candidatos;
	vector< cv::Rect > candidatos2;
	void filtrado_rois(cv::Mat& img);
	vector< cv::Mat > parametros(cv::Mat& img_binaria, vector<float>& extent,
			vector<float>& aspect_ratio);
	cv::Mat clothing_compensation();
	void mostrar_imagen();

};

#endif /* REGIONGROWING_H_ */
