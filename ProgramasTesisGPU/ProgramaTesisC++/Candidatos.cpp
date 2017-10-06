/*
 * Candidatos.cpp
 *
 *  Created on: Feb 2, 2017
 *      Author: luis
 */

#include "Candidatos.h"

Candidatos::Candidatos(){
	cout<<"Objeto candidatos se ha creado....."<<endl;
}

Candidatos::Candidatos(cv::Mat& img, Size dimensiones){
	met1 = HeadDetector(img);
	//(img, step, dimensiones)
	met2 = RegionGrowing(img, 0.05, dimensiones);
}

vector< cv::Mat > Candidatos::getCandidatosHeadDetector() {
	met1.detectarCabezas();
	return met1.getCandidatos();
}

vector< cv::Mat > Candidatos::getCandidatosRegionGrowing() {
	//Limitar los candidatos de este detector a 50

	met2.empezar_calculo();
	return met2.getCandidatos();
}

vector< cv::Mat > Candidatos::getCandidatosTodos() {
	vector<cv::Mat> candidatos1 = getCandidatosHeadDetector();
	vector<cv::Mat> candidatos2 = getCandidatosRegionGrowing();
	cout<<"candidatos1 "<<candidatos1.size()<<endl;
	cout<<"candidatos2 "<<candidatos2.size()<<endl;

	vector<cv::Mat> candidatosTotal;
	candidatosTotal.reserve(candidatos1.size() + candidatos2.size());
	candidatosTotal.insert( candidatosTotal.end(), candidatos1.begin(), candidatos1.end() );
	candidatosTotal.insert( candidatosTotal.end(), candidatos2.begin(), candidatos2.end() );
	cout<<"candidatosTotal "<<candidatosTotal.size()<<endl;
	return candidatosTotal;
}

// int main(){
//
//	 Candidatos regiones = Candidatos();
//	 Mat img = cv::imread("/home/luis/MEGA/Datasets/LSIFIR/Detection/Train/03/00400.png", CV_LOAD_IMAGE_COLOR);
//
//	 Mat imgF;
//	 img.convertTo(imgF, CV_32F); //increase the contrast (double)
//	 double min, max;
//	 cv::minMaxLoc(imgF, &min, &max);
//	 cout<<max<<" "<<min<<endl;
//	 float pendiente = (255.0) / (max - min);
//
//	 Mat resultado = pendiente * (imgF - max) + 255;
//	 cv::minMaxLoc(resultado, &min, &max);
//	 cout<<max<<" "<<min<<endl;
//
//	 resultado.convertTo(resultado, CV_8U);
//	 //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//	 imshow( "Display window", resultado);                   // Show our image inside it.
//
//	 waitKey(0);
//
//	 regiones.detectarCabezas(img);
//	 vector< Mat > rois = regiones.getCandidatos();
//
////	 for(int i=0; i<50; i++){
////		 //cout<<rois[i]<<endl;
////		 cout << "R (numpy)   = " << endl << cv::format(rois[i], 1) << endl << endl;
////		 //a = Rect(rois[i].at<int>(1),rois[i].at<int>(2),rois[i].at<int>(3),rois[i].at<int>(4));
////		 rectangle( img,
////		            Point( rois[i].at<int>(1), rois[i].at<int>(2)),
////		            Point( rois[i].at<int>(3), rois[i].at<int>(4)),
////		            Scalar( 0, 255, 255 ),
////		            1,
////		            8 );
////	 }
//
//	 return 0;
// }

