
''' Programa que prueba la deteccion usando image sustration
para determinar metricas en la deteccion. Se va a utilizar el conjunto
test de LSIFIR '''
import cv2
import sys
sys.path.insert(0, "/home/luis/fast-rcnn/caffe-fast-rcnn/python")
import caffe
import numpy as np

import math
class Clasificador:

    def __init__(self):
        self.__inicializar_red()
        self.__salida_red = None

    def __inicializar_red(self):
        ''' () -> ()

        Inicializa la red.

        '''
        #60000
        ruta_arq = "/home/luis/MEGA/red_caffe/ultima_red/"
        self.__net = caffe.Net(ruta_arq + 'test.prototxt',
                         ruta_arq + 'pesos2/_iter_60000.caffemodel',
                         caffe.TEST)

    def set_salida_red(self, datos_img, datos_grad, candidatos):
        ''' (numpy.ndarray, numpy.ndarray, numpy.ndarray, list) -> numpy.ndarray

        Alimenta la red con la imagen y las lista de candidatos.
        Devuelve la salida de la red, (#candidatos, 2)

        '''
        self.__net.blobs['data'].reshape(1, 2, int(math.ceil(129*1.4)), int(165*1.4))
        self.__net.blobs['rois'].reshape(len(candidatos), 5)

        # datos_img = cv2.resize(np.float32(cv2.imread(ruta_img_gray, 0) / 255.0), (165,129))
        # datos_grad = cv2.resize(gradiente(ruta_img_gray), (165,129))

        self.__net.blobs['data'].data[...] = np.asarray([datos_img, datos_grad])

        regiones = np.zeros((len(candidatos), 5))
        # print candidatos
        if len(candidatos) > 0:
            regiones[:, 1:5] = np.array(candidatos)

            self.__net.blobs['rois'].data[...] = regiones
            out = self.__net.forward()
            # print net.blobs['data'].data[0,0,:,:]
            # print out['cls_prob']
            # preds = np.argmax(out['cls_prob'], axis=1)
            # labels = net.blobs['labels'].data
            self.__salida_red = out['cls_prob']
        else:
            self.__salida_red = None

    def get_salida_red(self):
        return self.__salida_red
