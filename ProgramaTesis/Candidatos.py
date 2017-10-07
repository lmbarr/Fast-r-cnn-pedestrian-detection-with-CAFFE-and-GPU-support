import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from math import sqrt, ceil, floor
from numpy.matlib import repmat

class Candidatos:

    def __init__(self):
        self.__lista_candidatos = []

    def roi_umbral_dual_adaptativo(self, img_gray, dataset='LSIFIR'):
        ''' (numpy.ndarray, string) -> numpy.ndarray

        Calcula la img_binaria binarizada basada en aplicar un umbral dual para cada pixel
        Recive una img_gray en escala de grises uint8
        Devuelve una img_binaria binarizada

        '''

        img_gray = np.array(img_gray, dtype=np.float16)
        minimo = np.amin(img_gray)
        maximo = np.amax(img_gray)
        img_gray = 255 * (img_gray - minimo) / (maximo - minimo)
        img_gray = np.array(img_gray, dtype=np.int16)


        if dataset == 'LSIFIR':
            img_gray = cv2.resize(img_gray, (165,129))#(129, 164)
        elif  dataset == 'KMU':
            img_gray = cv2.resize(img_gray, (165,129))#(129, 164)
        elif dataset == 'OTCBVS':
            img_gray = cv2.resize(img_gray, (165,129))#(129, 164)

        se = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        w = 17
        alfa = 20
        tamano = (img_gray.shape[0], img_gray.shape[1], 2)
        img_binaria = np.zeros(img_gray.shape)

        for i in range(img_gray.shape[0]):
            for j in range(img_gray.shape[1]):
                acum = 0
                for k in range(i-w,i+w,1):
                    if k >= 0 and k < img_gray.shape[0]:
                        acum = img_gray[k, j] + acum

                tl = acum / (2 * w + 1) + alfa

                T3 = max(1.06 * (tl - alfa), tl + 2)
                T2 = min(T3, tl + 8)
                T1 = min(T2, 230)
                th = max(T1, tl)

                if img_gray[i, j] > th:
                    img_binaria[i, j] = 1
                elif img_gray[i ,j] < tl:
                    img_binaria[i, j] = 0
                elif (img_gray[i, j] <= th and img_gray[i, j] >= tl) and i > 0:
                    if img_binaria[i-1, j] == 1:
                        img_binaria[i, j] = 1
                    else:
                        img_binaria[i, j] = 0

        return cv2.morphologyEx(img_binaria, cv2.MORPH_CLOSE, se)

    def proyeccion_vertical_gradiente(self, img_gray):
        ''' (numpy.ndarray) -> numpy.ndarray

        Filtra la imagen generando franjas verticales dondes probablemten existen peatones
        Recive una img_gray en escala de grises uint8
        Devuelve una en escala de grises

        '''
        # Gradiente de la imagen
        grad = cv2.Laplacian(img_gray, cv2.CV_64F)
        # Aplicacion de un umbral
        _, img_binaria_grad = cv2.threshold(np.array(grad,dtype=np.int16),0,255,cv2.THRESH_BINARY)
        # Proyeccion vertical
        proyeccion_vertical = np.sum(img_binaria_grad, axis=0)
        # Suavizacion de la curva
        n = 6
        proyeccion_vertical_s = np.zeros(proyeccion_vertical.shape)

        for i in range(len(proyeccion_vertical)):
            acum = 0
            for k in range(i-int(floor(n/2.0)),i+int(ceil(n/2.0)),1):
                if k >= 0 and k < len(proyeccion_vertical):
                    acum = acum + proyeccion_vertical[k]

            proyeccion_vertical_s[i] = (1.0 / n) * acum

        # Calculo de Ts
        n = len(proyeccion_vertical_s)
        ux = np.sum(proyeccion_vertical_s) / n
        w = 0.8
        aux = np.sum(np.power(proyeccion_vertical_s - ux, 2))
        Ts = w * sqrt(aux / n)

        # Aplicar filtro a imagen
        th = proyeccion_vertical_s >= Ts
        t = repmat(th,img_binaria_grad.shape[0], 1)

        #plt.imshow(img_binaria_grad * t, cmap='gray')
        #plt.show()
        img_binaria_grad = img_binaria_grad * t
        return img_binaria_grad

    def detector_de_cabezas(self, img):
        ''' (numpy.ndarray) ->

        Mediante SURF detecta los cuerpos de la imagen y forma
        candidatos.
        Recive una imagen uint8 leida de opencv
        Devuelve una lista de BB en formato (x,y,w,h).

        '''

        height, width = img.shape[:2]
        img2 = cv2.resize(img,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

        surf = cv2.xfeatures2d.SURF_create(0.5,12,11,False,True)#(0.1,,,,)
        kp, _ = surf.detectAndCompute(img2, None)
        # img2 = cv2.drawKeypoints(img,kp,None,(0,0,0),flags=4)
        #############################################
        # Se forma el BB (x,y,w,h)
        h_base = int(ceil(129*1.4))
        w_base = int(165*1.4)
        h_img = 129.0
        w_img = 165.0

        escala_w = w_base / w_img
        escala_h = h_base / h_img
        BB = []
        if len(kp) != 0:
            for elemento in kp:
                if elemento.class_id < 0:

                    (x, y) = elemento.pt
                    x = x/3.0 - 1.0 * elemento.size/3.0 + 1
                    y = y/3.0 - 0.5 * elemento.size/3.0
                    w = 2 * elemento.size/3.0 #2
                    h = 4 * elemento.size/3.0 #3.5
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )
                        # print x,y,w,h
                    ###########################
                    (x, y) = elemento.pt
                    x = x/3.0 - 1.0 * elemento.size/3.0 + 1
                    y = y/3.0 - 0.5 * elemento.size/3.0
                    w = 2 * elemento.size/3.0 #2
                    h = 5 * elemento.size/3.0 #3.5
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )
                    #################################################
                    (x, y) = elemento.pt
                    x = x/3.0
                    y = y/3.0
                    diam = elemento.size/3.0
                    x = x - 0.5 * diam
                    y = y - 0.5 * diam
                    w = 1.5 * elemento.size/3.0 #2
                    h = 3.2 * elemento.size/3.0 #3.5
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )
                        # print x,y,w,h
                    ###########################
                    ###########################
                    (x, y) = elemento.pt
                    x = x/3.0 - 0.7 * elemento.size/3.0 + 1
                    y = y/3.0 - 0.3 * elemento.size/3.0
                    w = 2 * elemento.size/3.0 #2
                    h = 6 * elemento.size/3.0 #3.5
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )
                    #################################################
                    (x, y) = elemento.pt
                    h = elemento.size/3.0
                    w = h / 2.0
                    x = x/3.0 - w/2.0 + 1
                    y = y/3.0 - (elemento.size / 3) / 2.0
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )

                    (x, y) = elemento.pt
                    h = 25
                    w = 13
                    x = x/3 - 15
                    y = y/3 - 20/3
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )


                    (x, y) = elemento.pt
                    h = 35
                    w = 20
                    x = x/3.0 - 20
                    y = y/3.0 - 30
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )


                    (x, y) = elemento.pt
                    h = 45
                    w = 23
                    x = x/3.0 - 20
                    y = y/3.0 - 30
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )

                    (x, y) = elemento.pt
                    h = 52
                    w = 30
                    x = x/3.0 - 30
                    y = y/3.0 - 40
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )

                    (x, y) = elemento.pt
                    h = 55
                    w = 30
                    x = x/3.0 - 41
                    y = y/3.0 - 55
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )

                    (x, y) = elemento.pt
                    h = 20
                    w = 10
                    x = x/3.0 - 5
                    y = y/3.0 - 5
                    if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                        BB.append( (escala_w*x, escala_h*y, escala_w*(x+w), escala_h*(y+h)) )

        self.__lista_candidatos = BB

    def get_candidatos(self):
        return self.__lista_candidatos

    def cambiar_formato(self):
        ''' () -> list

        Recive una lista de candidatos
        Devuelve una lista de BB en formato (x1,y1,x2,y2).

        '''
        candidatos = self.__lista_candidatos
        self.__lista_candidatos = [[i[0],i[1],i[0]+i[2],i[1]+i[3]] for i in candidatos]
