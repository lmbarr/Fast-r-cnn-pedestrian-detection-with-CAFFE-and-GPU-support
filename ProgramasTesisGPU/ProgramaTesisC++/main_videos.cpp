/*
 * main_videos.cpp
 *
 *  Created on: Jun 9, 2017
 *      Author: luis
 */

#include "Detector.h"
#include "rutas.h"
int main(){
	// Definicion del video a usar
    string filename = RUTA_VIDEO;
    cv::VideoCapture capture(filename);
    Mat frame, img;

    // Definicion del tamaño de la imagen
    int w = 230, h = 181;
	Size dim = Size(w, h);

	Detector detector = Detector(dim);


    if( !capture.isOpened() )
        throw "Error de lectura";

    for( ; ; )
    {

    	//clock_t begin = clock();
    	capture >> frame;
        if(frame.empty())
            break;

        cv::cvtColor(frame, img, CV_BGR2GRAY);
    	clock_t begin = clock();
    	detector.empezarDeteccion(img, 0.92);

		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		cout<<"tiempo "<<elapsed_secs<<endl;
    }

    return 0;
}



