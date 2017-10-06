/*
 * HeadDetector.cpp
 *
 *  Created on: Apr 18, 2017
 *      Author: luis
 *  Pedestrian Detection in Far-Infrared Daytime Images Using a Hierarchical Codebook of SURF
 */

#include "HeadDetector.h"

HeadDetector::HeadDetector(){
	cout<<"Objeto HeadDetector creado...."<<endl;
}
HeadDetector::HeadDetector(cv::Mat& img) {
	// Recive la imagen img de 8 bits [0-255]
	cout<<"........"<<endl;
	this->img = img;
}

void HeadDetector::detectarCabezas(){
	int escala = 3;
	Size nueva_dim( Size(165 * escala, 129 * escala) );
	Mat dest;
	cv::resize(img, dest, nueva_dim, cv::INTER_CUBIC);

    int h_base = trunc(ceil(129*1.4));
    int w_base = trunc(165*1.4);
    float h_img = 129.0;
    float w_img = 165.0;
    float escala_w = w_base / w_img;
    float escala_h = h_base / h_img;



	//Definicion de SURF
	vector< KeyPoint > kp;
	//Este estoy usando (70,8,7) para los videos

	Ptr<SURF> detector = SURF::create(1, 8, 8, false, true);//(0.01, 21, 20, false, true)
	//0.1,11,12,false, true para lsifir
	detector->detect(dest, kp);

	vector< cv::Mat > BB;

    if(kp.size() > 0){
    	for(unsigned int i=0; i<kp.size(); i++){

    		if(kp[i].class_id < 0){

    			Point2f centro = kp[i].pt;
    			float x = centro.x; float y = centro.y;
                x = x / escala - 1.0 * kp[i].size / escala + 1;
                y = y / escala - 0.5 * kp[i].size / escala;
                float w = 2 * kp[i].size / escala;//2
                float h = 4 * kp[i].size / escala;//3.5

                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }

                centro = kp[i].pt;
                x = x / escala - 1.0 * kp[i].size / escala + 1;
                y = y / escala - 0.5 * kp[i].size / escala;
                w = 2 * kp[i].size / escala;//2
                h = 5 * kp[i].size / escala;//3.5

                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }

                centro = kp[i].pt;
                x = x / escala;
                y = y / escala;
                float diam = kp[i].size / escala;
                x = x - 0.5 * diam;
                y = y - 0.5 * diam;
                w = 1.5 * kp[i].size/escala;// #2
                h = 3.2 * kp[i].size/escala;// #3.5

                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }

                centro = kp[i].pt;
                x = x / escala - 0.7 * kp[i].size / escala + 1;
                y = y / escala - 0.3 * kp[i].size / escala;
                w = 2 * kp[i].size/escala; //#2
                h = 6 * kp[i].size/escala; //#3.5

                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }

                centro = kp[i].pt;
                h = kp[i].size/escala;
                w = h / 2.0;
                x = x/escala - w/2.0 + 1;
                y = y/escala - (kp[i].size / escala) / 2.0;

                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }

                centro = kp[i].pt;
                h = 25;
                w = 13;
                x = x/escala - 15;
                y = y/escala - 20.0/escala;
                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }

                centro = kp[i].pt;
                h = 35;
                w = 20;
                x = x/escala - 20;
                y = y/escala - 30;
                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }

                centro = kp[i].pt;
                h = 45;
                w = 23;
                x = x/escala - 20;
                y = y/escala - 30;

                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }

                centro = kp[i].pt;
                h = 52;
                w = 30;
                x = x/escala - 30;
                y = y/escala - 40;

                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }

                centro = kp[i].pt;
                h = 55;
                w = 30;
                x = x/escala - 41;
                y = y/escala - 55;

                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }

                centro = kp[i].pt;
                h = 20;
                w = 10;
                x = x/escala - 5;
                y = y/escala - 5;

                if((x + w) < img.size().width && (y + h) < img.size().height && x > 0 && y > 0){
                	BB.push_back((Mat_<float>(1,5) << 0, escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)));
                }
    		}
    	}
    	candidatos = BB;
    }else{
    	//candidatos = 0;
    	cout<<"No se detectaron bbs..."<<endl;
    }
}

vector< cv::Mat > HeadDetector::getCandidatos() {
	return candidatos;
}

