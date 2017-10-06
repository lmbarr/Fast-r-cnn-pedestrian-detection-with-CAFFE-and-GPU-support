/*
 * RegionGrowing.cpp
 *
 *  Created on: Apr 16, 2017
 *      Author: luis
 *
 * Clase que genera candidatos a peatones utilizando el metodo de region
 * growing descrito en el paper Detection of pedestrians in far-infrared automotive
 * night vision using region-growing and clothing distortion compensation.
 */

#include "RegionGrowing.h"

RegionGrowing::RegionGrowing(){
	cout<<"Objeto RegionGrowing creado..."<<endl;
	step = 0.1;
	umbral_inicial = 0.9;
}

RegionGrowing::RegionGrowing(cv::Mat& img, float step, Size dim) {
	// Recibe la imagen img de 8 bits [0-255]
	cv::resize(img, img, dim, cv::INTER_CUBIC);

	double min, max;
	Mat img_32F, img2;
	cv::minMaxLoc(img, &min, &max);
	double epsilon = 0.00001;

	if(max > 100){
		img.convertTo(img_32F, CV_32F, 1/255.0);
		cv::minMaxLoc(img_32F, &min, &max);
		this->img = ( ((1 - 0) /(max - min + epsilon)) * (img_32F - max) + 1) ;
	}else{
		img.convertTo(img_32F, CV_32F, 1/255.0);
	}

	this->umbral_inicial = max - 0.001;
	this->step = step;
}

cv::Mat RegionGrowing::clothing_compensation(){

    int morph_size = 4;
    Mat kernel = getStructuringElement( MORPH_RECT, Size( morph_size, morph_size ) );
    cv::Mat dst;
    if(!img.empty()){
    	cv::morphologyEx( img, dst, cv::MORPH_CLOSE, kernel );
    }
    return dst;
}

void RegionGrowing::mostrar_imagen(){

	for(unsigned int i = 0; i< candidatos.size(); i++ )
	 {
	   Scalar color = Scalar(0, 255, 0);
	   int x = candidatos[i].at<float>(1);
	   int y = candidatos[i].at<float>(2);
       int w = abs(candidatos[i].at<float>(1) - candidatos[i].at<float>(3));
       int h = abs(candidatos[i].at<float>(2) - candidatos[i].at<float>(4));
	   rectangle( img, Rect(x,y,w,h), color, 2, 8, 0 );
	 }

	/// Show in a window
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	imshow( "Contours", img );
    cv::waitKey (0);
}

void RegionGrowing::empezar_calculo(){

	img = clothing_compensation();

	cv::Mat img_binaria;

	for(float umbral=umbral_inicial; umbral>0.35; umbral=umbral-step){
		cout<<"umbrales "<<umbral<<endl;
    	cv::threshold(img, img_binaria, umbral, 1, cv::THRESH_BINARY);
    	filtrado_rois(img_binaria);
    }

    cout<<"num de candidatos "<<candidatos.size()<<endl;
    //mostrar_imagen();
}

vector< cv::Mat > RegionGrowing::getCandidatos(){
//	for(int i = 0; i < candidatos.size(); i++){
//		cout<<candidatos[i]<<endl;
//	}
	return candidatos;
}

void RegionGrowing::filtrado_rois(cv::Mat& img_binaria){
	/* Se ingresa la imagen binaria
       Calcula los bounding box que cumplen con el extend y el aspect ratio*/
	vector<float> extent, aspect_ratio;
	vector< cv::Mat > lista_rois = parametros(img_binaria, extent, aspect_ratio);
    // aspect_ratio[0-1] de rect vertical hasta cuadrado
    for(unsigned int j = 0; j < lista_rois.size(); j++){
        int w = abs(lista_rois[j].at<float>(1) - lista_rois[j].at<float>(3));
        int h = abs(lista_rois[j].at<float>(2) - lista_rois[j].at<float>(4));
        float area = (float) w * h;
        //cout<<w<<" "<<h<<endl;

//        cout<<"aspect ratio "<<aspect_ratio[j]<<endl;
//        cout<<"extent "<<extent[j]<<endl;

        if(aspect_ratio[j] > 0.2 && aspect_ratio[j] < 0.75
        		&& extent[j] > 0.25 && extent[j] < 0.9 && area > 100){
            candidatos.push_back(lista_rois[j]);
        }
    }
}


vector< cv::Mat > RegionGrowing::parametros(cv::Mat& img_binaria, vector<float>& extent,
		vector<float>& aspect_ratio){
	/* Devuelve todos los extent y aspect ratio de cada bb
	 * Devuelve tambien los bb(que forman el contorno) con el 0 agregado*/

	img_binaria.convertTo(img_binaria, CV_8U);
    int connectivity = 8;
    cv::Mat img_labels,stats,centroids;
    int numOfLables = cv::connectedComponentsWithStats(img_binaria, img_labels,
    		stats, centroids, connectivity);

	vector< cv::Mat > BB;
	vector< cv::Rect > boxes;

	double min, max;
	cv::minMaxLoc(img_labels, &min, &max);
	//cout<<min<<" "<<max<<endl;

    for (int j = 1; j < numOfLables; j++) {
            int area = stats.at<int>(j, CC_STAT_AREA);
            int x = stats.at<int>(j, CC_STAT_LEFT);
            int y  = stats.at<int>(j, CC_STAT_TOP);
            int w = stats.at<int>(j, CC_STAT_WIDTH);
            int h  = stats.at<int>(j, CC_STAT_HEIGHT);
        	extent.push_back((float)area / (w * h) );
        	aspect_ratio.push_back((float)w / h);// w/h
        	BB.push_back((Mat_<float>(1,5) << 0, x, y, x+w, y+h));
        	boxes.push_back(Rect(x, y, w, h));
    }

    return BB;
}
/*int main(){
	// Definicion del video a usar
    string filename = "/home/luis/Desktop/videos_infrarrojo1/video02.avi";
    cv::VideoCapture capture(filename);
    Mat frame, img;


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
    	RegionGrowing a = RegionGrowing(img);
    	a.empezar_calculo();

		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		cout<<"tiempo "<<elapsed_secs<<endl;
    }

    return 0;
}*/

//int main(){
//	string ruta_img = "/home/luis/MEGA/Datasets/CVCInfrared/NightTime/Train/FramesPos/000059.png";
//	Mat img = cv::imread(ruta_img, 0);
//	int w = 760, h = 440;
//	Size dim = Size(w, h);
//	RegionGrowing a = RegionGrowing(img, 0.01, dim);
//	a.empezar_calculo();
//	return 0;
//}
