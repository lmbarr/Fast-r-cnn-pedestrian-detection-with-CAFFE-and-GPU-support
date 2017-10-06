/*
 * Clasificador.h
 *
 *  Created on: Feb 6, 2017
 *      Author: luis
 */

#ifndef CLASIFICADOR_H_
#define CLASIFICADOR_H_

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

using namespace caffe;
using namespace std;
using namespace cv;

class Clasificador {

	public:
      void setSalidaRed(Mat &img, vector< Mat >* regiones);
      cv::Mat getSalidaRed();
      bool getSinCandidatos();
      Clasificador();
      Clasificador(cv::Size dim);

	private:
	  void apuntarCapaEntrada(vector<cv::Mat>* input_channels);
	  void apuntarCapaRoi(vector<cv::Mat>* input_roi, int numCandidatos);
	  void preprocesamiento(const Mat& img, vector<Mat>* input_channels);
      cv::Mat salidaRed;
      Net<float>* net;
      Size dimensiones;
      int canales;
};

#endif /* CLASIFICADOR_H_ */
