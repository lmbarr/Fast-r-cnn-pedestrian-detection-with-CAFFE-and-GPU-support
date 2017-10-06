/*
 * Detector.h
 *
 *  Created on: Feb 6, 2017
 *      Author: luis
 */

#ifndef DETECTOR_H_
#define DETECTOR_H_

#include "Candidatos.h"
#include "Clasificador.h"
#include  "NonMaximaSupression.h"
#include "opencv2/videoio.hpp"
#include <thread>
#include <future>
#include <chrono>
#include <atomic>
#include <sys/stat.h>
extern std::thread t;
void sonarAlarma();
extern std::atomic<bool> done;

class Detector {

	public:
		Detector(Size dim);
		void empezarDeteccion(Mat &img, float umbral);

	private:
		cv::Mat preprocesamiento_clasificador(Mat &img, Mat &grad);
		cv::Mat convertir_RGB(Mat &img);
		void mostrarImagen(Mat &resultado, vector< vector< Point > > rectangulos);
		cv::Mat calcularGradiente(Mat &img);
		cv::Mat preprocesamiento_img(Mat &img, Size dim);
		//void sonarAlarma();
		//std::thread t; //Definicion hilo alarma
		Candidatos candidato;
		Clasificador clasificador;
		Size dimensiones;
		cv::Mat salidaRed;
};

#endif /* DETECTOR_H_ */
