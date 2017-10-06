/*
 * Clasificador.cpp
 *
 *  Created on: Feb 2, 2017
 *      Author: luis
 */

#include "Clasificador.h"
#include "rutas.h"
// Definicion de funciones miembro

Clasificador::Clasificador() {
	cout<<"Crear objeto clasificador..........."<<endl;
	canales = 0;
	net = NULL;
}

Clasificador::Clasificador(Size dim) {

	cout<<"Crear objeto clasificador..........."<<endl;
	cout<<dim<<endl;
	string ruta_arq = RUTA_ARQUITECTURA;
	//Cargar red
	( MODO_CAFFE.compare("CPU") == 0 ) ? Caffe::set_mode(Caffe::CPU) : Caffe::set_mode(Caffe::GPU);
	net = new Net<float>(ruta_arq + "test.prototxt", caffe::TEST);
	net->CopyTrainedLayersFrom(ruta_arq + "_iter_60000.caffemodel");

	//(w,h)
	dimensiones = dim;
	canales = 2;
}

void Clasificador::apuntarCapaEntrada(vector<Mat >* input_channels) {
	Blob<float>* input_layer = net->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();

	//std::cout<<input_layer<<" "<<net->input_blobs()[0]<<std::endl;

	float* input_data = input_layer->mutable_cpu_data();

	for (int i = 0; i < input_layer->channels(); ++i) {
		Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Clasificador::apuntarCapaRoi(vector<Mat >* input_rois, int numCandidatos) {
	Blob<float>* input_layer = net->input_blobs()[1];

	cout<<"num candidatos "<<numCandidatos<<endl;
	float* input_data = input_layer->mutable_cpu_data();

	for (int i = 0; i < numCandidatos; ++i) {
		Mat channel(1, 5, CV_32FC1, input_data);
		input_rois->push_back(channel);
		input_data += 5;
	}
}

void Clasificador::preprocesamiento(const Mat& img, vector<Mat>* input_channels) {
      cv::split(img, *input_channels);
}

void Clasificador::setSalidaRed(Mat &img, vector<Mat>* regiones){
	if(regiones->size() > 0){
//		cout<<"canal candidato "<<regiones->size()<<endl;
//		cv::merge(*regiones, candidatos);
//		cout<<"canal candidato "<<regiones->size()<<endl;
//		cout<<"canal candidato "<<candidatos.channels()<<endl;

		//Reshape blob de entrada imagen

		Blob<float>* input_layer = net->input_blobs()[0];
		input_layer->Reshape(1, canales, dimensiones.height, dimensiones.width);

		//Reshape blob de rois
		Blob<float>* input_rois_layer = net->input_blobs()[1];
		vector<int> shape(2); shape[0] = regiones->size(); shape[1] = 5;
		cout<<input_rois_layer->shape_string()<<endl;
		input_rois_layer->Reshape(shape);
		cout<<input_rois_layer->shape_string()<<endl;

		/* Forward dimension change to all layers. */
		net->Reshape();
		vector<Mat> input_channels, input_rois;
		apuntarCapaEntrada(&input_channels);
		apuntarCapaRoi(&input_rois, regiones->size());
		preprocesamiento(img, &input_channels);


		for(unsigned int i=0;i<regiones->size(); i++){
			(*regiones)[i].copyTo(input_rois[i]);
		}




//		std::cout<<"size vector "<<input_rois.size()<<std::endl;
//		std::cout<<net->input_blobs()[1]->data_at(0,0,0,1)<<std::endl;
//		std::cout<<net->blob_by_name("rois")->data_at(0,4,0,0)<<std::endl;
//		std::cout<<net->blob_by_name("rois")->shape_string()<<std::endl;


		float* loss=0;
		net->ForwardPrefilled(loss);

		/* Copy the output layer to a std::vector */
		Blob<float>* output_layer = net->output_blobs()[0];

		float* begin = output_layer->mutable_cpu_data();

		cout<<output_layer->num()<<endl;
		cout<<output_layer->channels()<<endl;
		cout<<output_layer->shape_string()<<endl;



		salidaRed = Mat(output_layer->num(), output_layer->channels(), CV_32F, begin);

	}else{
		cout<<"No existen candidatos..."<<endl;
	}
}

cv::Mat Clasificador::getSalidaRed(){
	return salidaRed;
}


//int main(int argc, char** argv) {
////Estoy pasando los argumentos en la configuracion de run
//  ::google::InitGoogleLogging(argv[0]);
//
//  cv::Mat img = cv::imread("/home/luis/MEGA/Datasets/LSIFIR/Detection/Train/03/00400.png", 0);
//  cv::Mat img2 = img;
//  std::vector<cv::Mat> Canales;
//  Canales.push_back(img);
//  Canales.push_back(img2);
//  cv::Mat img_Total;
//  cv::merge(Canales, img_Total);
//  Size b = Size(145, 150);
//  Clasificador a = Clasificador(b);
//  float dummy_query_data[10] = { 0, 0, 3, 4, 5, 6, 70, 80, 70, 100 };
//  //cv::Mat candidatos = cv::Mat(1, 5, CV_32FC2);
////  cout<<candidatos.channels()<<endl;
////  cout << "M = " << endl << " " << candidatos << endl << endl;
////  cout << "R (numpy)   = " << endl << cv::format(candidatos, 1) << endl << endl;
////  std::vector<cv::Mat> candidatos;
////  a.setSalidaRed(img_Total, &candidatos);
//}
