/*
 * NonMaximaSupression.cpp
 *
 *  Created on: Feb 7, 2017
 *      Author: luis
 */

#include "NonMaximaSupression.h"

MiEstructura::MiEstructura(float confidencia, cv::Mat& bb){
	this->confidencia = confidencia;
	this->bb = bb;
}

float intersection(Mat & elemento1, Mat & elemento2){
	float xA1 = elemento1.at<float>(0);
	float yA1 = elemento1.at<float>(1);
	float xA2 = elemento1.at<float>(2);
	float yA2 = elemento1.at<float>(3);
	//(xA1, yA1, xA2, yA2) = elemento1
	float xB1 = elemento2.at<float>(0);
	float yB1 = elemento2.at<float>(1);
	float xB2 = elemento2.at<float>(2);
	float yB2 = elemento2.at<float>(3);
	//(xB1, yB1, xB2, yB2) = elemento2
	float valor = std::max((float)0.0, std::min(xA2, xB2) - std::max(xA1, xB1)) *
			std::max((float)0.0, std::min(yA2, yB2) - std::max(yA1, yB1));
	return valor;
}

float unionn(Mat & elemento1, Mat & elemento2, float intersectionArea){
	float xA1 = elemento1.at<float>(0);
	float yA1 = elemento1.at<float>(1);
	float xA2 = elemento1.at<float>(2);
	float yA2 = elemento1.at<float>(3);
	//(xA1, yA1, xA2, yA2) = elemento1
	float xB1 = elemento2.at<float>(0);
	float yB1 = elemento2.at<float>(1);
	float xB2 = elemento2.at<float>(2);
	float yB2 = elemento2.at<float>(3);
	//(xB1, yB1, xB2, yB2) = elemento2

	float w1 = max(xA1, xA2) - min(xA1, xA2);
	float h1 = max(yA1, yA2) - min(yA1, yA2);
	float w2 = max(xB1, xB2) - min(xB1, xB2);
	float h2 = max(yB1, yB2) - min(yB1, yB2);
	return (w1*h1 + w2*h2 - intersectionArea);
}

float IoU(Mat & elemento1, Mat & elemento2){
	float interseccion = intersection(elemento1, elemento2);
	if(interseccion / unionn(elemento1, elemento2, interseccion) == 0)
		return 0;
	else
		return interseccion / unionn(elemento1, elemento2, interseccion);
}
bool less_than_key(const MiEstructura& struct1, const MiEstructura& struct2)
{
	return (struct1.confidencia > struct2.confidencia);
}
vector<vector<Point> > non_max_suppression(Mat &lista_bb, vector<float>* confidencias, float solapamiento=0.5){
	/* Recive dos listas, confidencias es una lista de probablialdades
	* y lista_bb es las lista de predetecciones (x1,y1,x2,y2)
	* Hay correspondencias entre ambas listas*/

	vector < MiEstructura > lista_compuesta;
	for(unsigned int i = 0; i < confidencias->size(); i++){
		Mat roi = lista_bb.row(i);
		lista_compuesta.push_back(MiEstructura((*confidencias)[i], roi));
	}
	std::sort(lista_compuesta.begin(), lista_compuesta.end(), less_than_key);
//////////////impresion//////////////////
//	cout<<"impresion "<<endl;
//	for(unsigned int i = 0; i < lista_compuesta.size(); i++){
//		cout<<lista_compuesta[i].confidencia;
//		cout<<lista_compuesta[i].bb<<endl;
//	}
////////////////////////////////////////
    vector<cv::Mat> G;
    vector< float > probs;
    //Copia en una nueva posicion de memoria
    vector < MiEstructura >  lista_compuesta_copia = lista_compuesta;

    vector< float > lista_solapamiento;
    for(unsigned int j = 0; j < lista_compuesta_copia.size(); j++){
    	MiEstructura elemento = lista_compuesta_copia[j];
        if(G.size() == 0){
//        	cout<<"G esta vacio "<<elemento.confidencia;
//        	cout<<elemento.bb<<endl;
            G.push_back(elemento.bb);
            probs.push_back(elemento.confidencia);
            lista_compuesta.erase(std::remove(lista_compuesta.begin(), lista_compuesta.end(),
        			elemento), lista_compuesta.end());
            continue;
        }
        lista_solapamiento = vector< float >();
        for(unsigned int i = 0; i < G.size(); i++){
            // x1,y1,x2,y2
            lista_solapamiento.push_back(IoU(elemento.bb, G[i]));
        }
////Impresion de lista_solapamiento//////////////
//    	cout<<"Impresion de lista_solapamiento "<<endl;
//    	for(unsigned int k = 0; k < lista_solapamiento.size(); k++){
//    		cout<<lista_solapamiento[k]<<" " ;
//    	}
/////////////////////////////////////////////////
        float valor_maximo = *std::max_element(lista_solapamiento.begin(), lista_solapamiento.end());

        if(valor_maximo > solapamiento){
        	lista_compuesta.erase(std::remove(lista_compuesta.begin(), lista_compuesta.end(),
        			elemento), lista_compuesta.end());
        }
        else{
        	G.push_back(elemento.bb);
        	probs.push_back(elemento.confidencia);
        	lista_compuesta.erase(std::remove(lista_compuesta.begin(), lista_compuesta.end(),
        	       elemento), lista_compuesta.end());
        }
    }
    ////Impresion de G//////////////
	//cout<<"Impresion de G "<<endl;
//	for(unsigned int i = 0; i < G.size(); i++){
//		cout<<probs[i];
//		cout<<G[i]<<endl;
//	}
	///Se pasa a la forma de rectangulos con 4 puntos::
    Mat rois, temp;
	vector<vector<Point> > rectangulos;
	vector<Point> elemento(4);

    for(unsigned int k = 0; k < G.size(); k++){
    	temp = G[k];
		Point p0(temp.at<float>(0), temp.at<float>(1));
		Point p1(temp.at<float>(0), temp.at<float>(3));
		Point p2(temp.at<float>(2), temp.at<float>(3));
		Point p3(temp.at<float>(2), temp.at<float>(1));
		//cout<<p0<<p1<<p2<<p3<<endl;
		elemento[0]=p0;elemento[1]=p1;elemento[2]=p2;elemento[3]=p3;
		rectangulos.push_back(elemento);
    }
    return rectangulos;
}


// Malisiewicz et al.
vector<vector<Point> > non_max_suppression_fast(Mat &boxes, float overlapThresh=0.4){

    /*[x1 x2 y1 y2
     * x1 x2 y1 y2
     * x1 x2 y1 y2]
     * Retorna un vector de contornos de 4 puntos que forman
     * el rectangulo*/

    // initialize the list of picked indexes
    vector< int > pick;

    // grab the coordinates of the bounding boxes
    Mat x1 = boxes.col(0).clone();
    Mat y1 = boxes.col(1).clone();
    Mat x2 = boxes.col(2).clone();
    Mat y2 = boxes.col(3).clone();
//    cout<<"x1 "<<x1<<endl;
//    cout<<"y1 "<<y1<<endl;
//    cout<<"x2 "<<x2<<endl;
//    cout<<"y2 "<<y2<<endl;

//    # compute the area of the bounding boxes and sort the bounding
//    # boxes by the bottom-right y-coordinate of the bounding box
    Mat area = (x2 - x1 + 1).mul(y2 - y1 + 1);
    Mat idxs;
//    cout<<"area "<<area<<endl;
    cv::sortIdx(y2, idxs, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
//    cout<<"idxs "<<idxs<<endl;
    // keep looping while some indexes still remain in the indexes
    // list
    int last, i;
    while(idxs.rows > 0){
//    	# grab the last index in the indexes list and add the
//    	# index value to the list of picked indexes
    	last = idxs.rows - 1;
//    	cout<<"last "<<last<<endl;
//    	cout<<idxs.type()<<endl;
    	i = idxs.at<int>(last);
//    	cout<<"i "<<i<<endl;
    	pick.push_back(i);

//    	# find the largest (x, y) coordinates for the start of
//    	# the bounding box and the smallest (x, y) coordinates
//    	# for the end of the bounding box
    	Mat aux1 = idxs.rowRange(0, last).clone();
//    	cout<<"aux1 "<<aux1<<endl;
    	Mat xx1_,yy1_, xx2_, yy2_;


    	for(int k=0; k<aux1.rows; k++){
    		//cout<<x2.row(aux1.at<int>(k))<<endl;
    		xx1_.push_back(x1.row(aux1.at<int>(k)));
    		yy1_.push_back(y1.row(aux1.at<int>(k)));
    		xx2_.push_back(x2.row(aux1.at<int>(k)));
    		yy2_.push_back(y2.row(aux1.at<int>(k)));
    	}

    	Mat xx1, xx2, yy1, yy2;
    	cv::max(xx1_, x1.at<float>(i), xx1);
    	cv::max(yy1_, y1.at<float>(i), yy1);
    	cv::min(xx2_, x2.at<float>(i), xx2);
    	cv::min(yy2_, y2.at<float>(i), yy2);

//    	cout<<"xx1 "<<xx1<<endl;
//    	cout<<"yy1 "<<yy1<<endl;
//    	cout<<"xx2_"<<xx2_<<endl;
//    	cout<<"yy2 "<<yy2<<endl;

//    	# compute the width and height of the bounding box
    	Mat w, h;
    	cv::max(xx2 - xx1 + 1, 0, w);
    	cv::max(yy2 - yy1 + 1, 0, h);
//    	cout<<"W "<<w<<endl;
//    	cout<<"H "<<h<<endl;

//    	# compute the ratio of overlap
    	Mat area_, overlap;
    	for(int k=0; k<aux1.rows; k++){
    		area_.push_back(area.row(aux1.at<int>(k)));
    	}
    	cv::divide(w.mul(h), area_, overlap);

    	//cout<<area_<<endl;
    	//cout<<"overlap "<<overlap<<endl;


//    	# delete all indexes from the index list that have
    	Mat aux;
    	aux = overlap > overlapThresh;
    	//cout<<"aux "<<aux<<endl;
    	Mat indices;
    	cv::findNonZero(aux, indices);

    	indices.push_back(Point(0, last));
    	Mat idxs1 = idxs;
    	idxs.release();

    	bool control = false;
//    	cout<<"indices "<<indices<<endl;
//        cout<<indices.at<Point>(0).y<<endl;
//        cout<<indices.at<Point>(1).y<<endl;
//        cout<<indices.at<Point>(2).y<<endl;
//        cout<<indices.at<Point>(3).y<<endl;

    	for(int i=0;i<idxs1.rows; i++){
    		for(int j=0; j<indices.rows; j++){
    			if(i == indices.at<Point>(j).y){
    				//cout<<"i "<<i<<endl;
    				control = true;
    			}
    		}
			if(!control){
				idxs.push_back(idxs1.at<int>(i));
   			}
			control = false;
    	}
    	//cout<<"idxs"<<idxs<<endl;

    	//idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    }

    //return boxes[pick].astype("int").tolist();
    //for(int i=0; i<pick.size();i++) cout<<pick[i]<<endl;
    Mat rois, temp;
	vector<vector<Point> > rectangulos;
	vector<Point> elemento(4);

    for(unsigned int i=0; i<pick.size(); i++){
    	rois.push_back(boxes.row(pick[i]));
    	temp = boxes.row(pick[i]);
		Point p0(temp.at<float>(0), temp.at<float>(1));
		Point p1(temp.at<float>(0), temp.at<float>(3));
		Point p2(temp.at<float>(2), temp.at<float>(3));
		Point p3(temp.at<float>(2), temp.at<float>(1));
		elemento[0]=p0;elemento[1]=p1;elemento[2]=p2;elemento[3]=p3;
		rectangulos.push_back(elemento);
    }
    return rectangulos;
}


/*int main(){
	Mat rois = (Mat_<float>(7,4) << 11, 5, 30, 70,
									9, 5, 30, 71,
								  500, 10, 550, 40,
								  20, 25, 35, 40,
								  20, 19, 30, 40,
								  16,15,15,20,
								  100, 50, 150, 100);
	cout << "R (numpy)   = " << endl << format(rois,1 ) << endl << endl;
	//rois.convertTo(rois, CV_8U);
	cout<<rois.type()<<endl;
	vector<float> confidencias;
	confidencias.push_back(0.188);
	confidencias.push_back(0.3444);
	confidencias.push_back(0.7);
	confidencias.push_back(0.5);
	confidencias.push_back(0.2);
	confidencias.push_back(0.4);
	confidencias.push_back(0.657);
	cout<<confidencias[2]<<endl;
	non_max_suppression(rois, &confidencias, 0.5);

//	non_max_suppression_fast(rois);
	return 0;
}*/
