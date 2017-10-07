from Candidatos import Candidatos
from Clasificador import Clasificador
from matplotlib import pyplot as plt
import cv2
import numpy as np
import sys
import math

sys.path.append('/home/luis/MEGA/red_caffe/red1/evaluacion_red_rois')
from roi_helpers import non_max_suppression_fast
import matplotlib.patches as patches
from non_maxima_supresion import non_maxima_supression

class Detector:

    def __init__(self):
        self.__candidatos = Candidatos()
        self.__clasificador = Clasificador()
        self.__BB_final = []
        self.__pro_seleccionados = []
        plt.ion()

    def gradiente(self, img):
        # img = cv2.imread(ruta_img, 0)
        # img = np.float32(img) / 255.0

        # Calculate gradient
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
        angle = angle / (2 * 3.1415912)
        return angle

    def empezar_deteccion(self, img, umbral=0.85):
        ''' fshdfhdf

        '''
        ####Procesar imagen para clase Candidatos########
        # img_candidatos = cv2.resize(np.float32(img) / 255.0, (165, 129))
        # img_candidatos = ((255.0 - 0) /( np.amax(img_candidatos) - np.amin(img_candidatos)) ) * \
        # ( img_candidatos - np.amax(img_candidatos) ) + 255.0
        # img_candidatos = np.uint8(img_candidatos)

        ########################################
        self.__candidatos.detector_de_cabezas(cv2.resize(img, (165, 129)))
        # self.__candidatos.cambiar_formato()
        regiones = self.__candidatos.get_candidatos()

        img = cv2.resize(np.float32(img) / 255.0, (int(165*1.4), int(math.ceil(129*1.4))))
        epsilon = 0.000001
        img = ((1 - 0) / ( np.amax(img) - np.amin(img) + epsilon) ) * ( img - np.amax(img) ) + 1

        grad = self.gradiente(img)
        self.__BB_crudo = regiones
        self.__clasificador.set_salida_red(img, grad, regiones)
        salida_red = self.__clasificador.get_salida_red()
        self.__salida_red = salida_red

        BB_seleccionados = []
        pro_seleccionados = []


        # fig = plt.figure(1, figsize=(10,10))
        # ax1 = fig.add_subplot(111)
        # ax1.imshow(cv2.resize(img, (int(165*1.4),int(math.ceil(129*1.4))) ) ,'gray')

        if salida_red == None:
            pass
        else:
            for k in range(salida_red.shape[0]):
                if salida_red[k, 1] > umbral:
                  x1,y1,x2,y2 = regiones[k]
                  BB_seleccionados.append([x1,y1,x2,y2])
                  pro_seleccionados.append(salida_red[k, 1])


        if len(BB_seleccionados) > 0:
            # self.__BB_final = BB_seleccionados
            self.__pro_seleccionados = pro_seleccionados
            self.__BB_final = non_max_suppression_fast(BB_seleccionados, 0.4)
            # print len(self.__BB_final)
            # self.__BB_final, self.__pro_seleccionados = non_maxima_supression(pro_seleccionados,
                # BB_seleccionados, solapamiento=0.4)

        #     for elemento in self.__BB_final:
        #         x = elemento[0]
        #         y = elemento[1]
        #         w = abs(elemento[2]-elemento[0])
        #         h = abs(elemento[3]-elemento[1])
        #         rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='g',facecolor='none')
        #         ax1.add_patch(rect)
        #
        #
        # plt.pause(0.001)
        # plt.draw()
        # plt.clf()

    def get_BB_final(self):
        return self.__BB_final

    def get_BB_crudo(self):
        return self.__BB_crudo

    def get_salida_red_cruda(self):
        return self.__salida_red
    def get_pro(self):
        return self.__pro_seleccionados
