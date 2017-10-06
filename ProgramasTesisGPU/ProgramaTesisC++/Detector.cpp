/*
 * Detector.cpp
 *
 *  Created on: Feb 5, 2017
 *      Author: luis
 */

#include "Detector.h"
#include "rutas.h"

std::thread t;
std::atomic<bool> done(false);
bool primeraVez = true;

Detector::Detector(Size dim) {
   cout << "Se ha creado un objeto Detector.........." << endl;
   clasificador = Clasificador(dim);
   dimensiones = dim;
}

void Detector::empezarDeteccion(Mat &img, float umbral=0.5) {
	//La imagen entra directamente con su tamaño original
	///////////////////////////////////////////////
	Mat img_candidatos;
	cv::resize(img, img_candidatos, Size(165, 129));
	candidato = Candidatos(img_candidatos, dimensiones);
	vector< Mat > regiones = candidato.getCandidatosTodos();
	//vector< Mat > regiones = candidato.getCandidatosRegionGrowing();
	//vector< Mat > regiones = candidato.getCandidatosHeadDetector();
	cout<<"numero de candidatos que entran en red "<<regiones.size()<<endl;
	cout<<"entro en procesamiento "<<endl;
	//////////////////////////////////////////////

	cout<<"img tamaño original "<<img.size()<<endl;
	Mat img_red = preprocesamiento_img(img, dimensiones);
	Mat grad = calcularGradiente(img_red);

//	vector<Mat> regiones;
//	float dummy_query_data[5] = { 0,1, 50,114, 100};
//	cv::Mat candidatos = cv::Mat(1, 5, CV_32FC1,dummy_query_data);
//	regiones.push_back(candidatos);

	Mat imgT = preprocesamiento_clasificador(img_red, grad);
	clasificador.setSalidaRed(imgT, &regiones);
	salidaRed = clasificador.getSalidaRed();

	Mat roi, boxes;
	vector<vector<Point> > rectangulos;
	vector<float> pro_seleccionados;
    //vector<Point> elemento(4);


    if(regiones.size() > 0){
    	for(int i=0; i<salidaRed.rows; i++){
    		if(salidaRed.at<float>(i, 1) >= umbral){
    			roi = regiones[i];
    			//toma una submatriz, le quita el cero
    			boxes.push_back(Mat(roi, Range::all(), Range(1,5)));
    		    pro_seleccionados.push_back(salidaRed.at<float>(i, 1));

/*    			Point p0(roi.at<float>(1), roi.at<float>(2));
    			Point p1(roi.at<float>(1), roi.at<float>(4));
    			Point p2(roi.at<float>(3), roi.at<float>(4));
    			Point p3(roi.at<float>(3), roi.at<float>(2));
    			elemento[0]=p0;elemento[1]=p1;elemento[2]=p2;elemento[3]=p3;
    			rectangulos.push_back(elemento);*/

    		}
    	}
    }

    if(boxes.rows > 0){
    	rectangulos= non_max_suppression_fast(boxes, 0.4);
    	cout<<"Candidatos despues de NMS "<<rectangulos.size()<<endl;
    }

    if(rectangulos.size() > 0){
    	if(primeraVez){
    		t = std::thread(sonarAlarma);
    		primeraVez = false;
    	}

    	if(done){

			if(t.joinable()){
				t.detach();
			}
			done = false;
    		t = std::thread(sonarAlarma);

    	}
    }

    Mat resultado = convertir_RGB(img_red);
    mostrarImagen(resultado, rectangulos);
}

void sonarAlarma(){
	/* */
	string filename  = RUTA_ALARMA;
	string command = "aplay -c 1 -q -t wav " + filename;
	/* play sound */
	system( command.c_str() );
	done = true;
}

void Detector::mostrarImagen(Mat &resultado, vector< vector< Point > > rectangulos) {
	/* Dibuja los contornos en la imagen y muestra la imagen.*/
    cv::drawContours(resultado, rectangulos, -1, Scalar(0, 255, 0), 2.5);
    cv::imshow( "Display window", resultado );
    cv::waitKey (1);
}

cv::Mat Detector::preprocesamiento_clasificador(Mat &img, Mat &grad) {

	/* Entra la imagen y el gradiente del mismo tamaño.
	 * Retorna la imagen compuesta de estos dos canales.*/

	vector< Mat > imgV;
	imgV.push_back(img);
	imgV.push_back(grad);

	Mat imgTotal;
	cv::merge(imgV, imgTotal);

	return imgTotal;

}

cv::Mat Detector::convertir_RGB(Mat &img) {
	/*Aumenta el constraste de la imagen y la convierte a rgb.
	 *Para dibujar el rectangulo color verde
	 *Devuelve una imagen rgb de 3 canales*/
	Mat rgb;
	cvtColor(img, rgb, CV_GRAY2BGR);
	return rgb;
}
cv::Mat Detector::preprocesamiento_img(Mat &img, Size dim) {
	/* Entra la imagen[0-255] leida de imread y con tamaño original.
	 * Devuelve la imagen en todo su espectro cubre[0-1] lista para
	 * entrar en la red*/

	double min, max;
	Mat img_32F, img2;
	cv::minMaxLoc(img, &min, &max);
	//cout<<"min "<<min<<" max "<<max<<endl;
	double epsilon = 0.00001;

	if(max > 100){
		img.convertTo(img_32F, CV_32F, 1/255.0);
		cv::resize(img_32F, img2, dim);
		cv::minMaxLoc(img2, &min, &max);
		return ( ((1 - 0) /(max - min + epsilon)) * (img2 - max) + 1) ;
	}else{
		img.convertTo(img_32F, CV_32F, 1/255.0);
		cv::resize(img_32F, img2, dim);
		return img2;
	}
}

cv::Mat Detector::calcularGradiente(Mat &img) {
	/*  Recive la imgen de procesamiento_img, la cual cubre todo
	 *  su espectro[0-1]
	 *  Devuelve el gradiente de esta imagen con el mismo tamaño que entro*/

	Mat gx, gy;
	cv::Sobel(img, gx, CV_32F, 1, 0, 1);
	cv::Sobel(img, gy, CV_32F, 0, 1, 1);
	Mat mag, angle;
	cv::cartToPolar(gx, gy, mag, angle);
	angle.convertTo( angle, CV_32F, 1/(2 * 3.14159) );
    return angle;
}


