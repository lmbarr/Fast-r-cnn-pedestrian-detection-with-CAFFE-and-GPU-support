/*
 * main_LSI.cpp
 *
 *  Created on: Jun 9, 2017
 *      Author: luis
 */

/*#include "Detector.h"
#include "rutas.h"

bool fexists(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  return (bool)ifile;
}

inline bool exists_test3 (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

int main() {

    int w = 230, h = 181;
	Size dim = Size(w, h);

    cv::namedWindow( "Display window", WINDOW_NORMAL );
    cv::resizeWindow("Display window", w, h);
	std::ostringstream s1;
	string idx, escena, ruta;
	string ruta_img = RUTA_FOLDER_LSI;

	Detector detector = Detector(dim);

	for(int j=1; j<7; j++){
		for(int i=1; i<1000; i++){
			s1 << std::setw( 5 ) << std::setfill( '0' ) << i;
			idx = s1.str();s1.str("");
			s1 << std::setw( 2 ) << std::setfill( '0' ) << j;
			escena = s1.str();s1.str("");
			ruta = ruta_img + escena + "/" + idx + ".png";

			clock_t begin = clock();

	        if(!fexists(ruta)){
	            continue;
	        }


	    	Mat img = imread(ruta, 0);

		cout<<img.empty()<<endl;
		cout<<ruta<<endl;
	    	detector.empezarDeteccion(img, 0.6);

	    	clock_t end = clock();
	    	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	        cout<<"tiempo "<<elapsed_secs<<endl;
		}
	}

	return 0;
}*/
