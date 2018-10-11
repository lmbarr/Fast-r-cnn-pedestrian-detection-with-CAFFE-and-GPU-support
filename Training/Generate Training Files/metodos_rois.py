import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from math import sqrt, ceil, floor
from numpy.matlib import repmat

def roi_movimiento_frame(img_gray_t0, img_gray_t1, dataset='LSIFIR'):
    ''' Calcula la img_binaria binarizada basado en la resta de dos frames consecutivos
    Recive dos imagenes de uint8
    Devuelve una img_binaria binaria'''

    img_gray_t0 = np.array(img_gray_t0, dtype=np.int16)
    img_gray_t1 = np.array(img_gray_t1, dtype=np.int16)

    if dataset == 'LSIFIR':
        pass
    elif  dataset == 'KMU':
        img_gray_t0 = cv2.resize(img_gray_t0, (164,129))#(129, 164)
        img_gray_t1 = cv2.resize(img_gray_t1, (164,129))#(129, 164)
    elif dataset == 'OTCBVS':
        img_gray_t0 = cv2.resize(img_gray_t0, (164,129))#(129, 164)
        img_gray_t1 = cv2.resize(img_gray_t1, (164,129))#(129, 164)

    img_restada = img_gray_t1 - img_gray_t0
    # print np.amax(img_restada), np.amin(img_restada)
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 15))
    img_binaria = np.zeros(img_restada.shape, dtype=np.uint8)

    #_, img_binaria = cv2.threshold(img_restada, 0, 1, cv2.THRESH_BINARY)

    for k in range(img_restada.shape[0]):
        for l in range(img_restada.shape[1]):
            if img_restada[k, l] > 0:
                img_binaria[k, l] = 1
            else:
                img_binaria[k, l] = 0
    #Closing de la img_binaria
    img_closed = cv2.morphologyEx(img_binaria, cv2.MORPH_CLOSE, se)

    return img_closed

def roi_umbral_dual_adaptativo(img_gray, dataset='LSIFIR'):
    '''Calcula la img_binaria binarizada basada en aplicar un umbral dual para cada pixel
    Recive una img_gray en escala de grises uint8
    Devuelve una img_binaria binarizada'''

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


def proyeccion_vertical_gradiente(img_gray):
    '''Filtra la imagen generando franjas verticales dondes probablemten existen peatones
    Recive una img_gray en escala de grises uint8
    Devuelve una en escala de grises '''

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




def detector_de_cabezas(img):
    '''Mediante SURF detecta las cabezas de los peatones y arma
    el BB.
    Devuelve el una lista de BB en formato (x1,y1,x2,y2).
    '''
    # img = cv2.imread(ruta,0)
    # img1 = mpimg.imread(ruta)

    height, width = img.shape[:2]
    img2 = cv2.resize(img,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

    surf = cv2.xfeatures2d.SURF_create(0.4,9,9,False,True)#(2,,,,)
    kp, _ = surf.detectAndCompute(img2, None)
    # img2 = cv2.drawKeypoints(img,kp,None,(0,0,0),flags=4)
    #############################################
    # Se forma el BB (x,y,w,h)
    BB = []
    if len(kp) != 0:
        for elemento in kp:

            (x, y) = elemento.pt
            x = x/3.0 - 1.0 * elemento.size/3.0 + 1
            y = y/3.0 - 0.5 * elemento.size/3.0
            w = 2 * elemento.size/3.0 #2
            h = 4 * elemento.size/3.0 #3.5
            if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                BB.append((x, y, w, h))
                # print x,y,w,h
            ###########################
            (x, y) = elemento.pt
            x = x/3.0 - 1.0 * elemento.size/3.0 + 1
            y = y/3.0 - 0.5 * elemento.size/3.0
            w = 2 * elemento.size/3.0 #2
            h = 5 * elemento.size/3.0 #3.5
            if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                BB.append((x, y, w, h))
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
                BB.append((x, y, w, h))
                # print x,y,w,h
            ###########################
            ###########################
            (x, y) = elemento.pt
            x = x/3.0 - 0.7 * elemento.size/3.0 + 1
            y = y/3.0 - 0.3 * elemento.size/3.0
            w = 2 * elemento.size/3.0 #2
            h = 6 * elemento.size/3.0 #3.5
            if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                BB.append((x, y, w, h))
            #################################################
            (x, y) = elemento.pt
            h = elemento.size/3.0
            w = h / 2.0
            x = x/3.0 - w/2.0 + 1
            y = y/3.0 - (elemento.size / 3) / 2.0
            if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                BB.append((x, y, w, h))

            (x, y) = elemento.pt
            h = 25
            w = 13
            x = x/3 - 15
            y = y/3 - 20/3
            if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                BB.append((x, y, w, h))


            (x, y) = elemento.pt
            h = 35
            w = 20
            x = x/3.0 - 20
            y = y/3.0 - 30
            if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                BB.append((x, y, w, h))


            (x, y) = elemento.pt
            h = 45
            w = 23
            x = x/3.0 - 20
            y = y/3.0 - 30
            if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                BB.append((x, y, w, h))

            (x, y) = elemento.pt
            h = 52
            w = 30
            x = x/3.0 - 30
            y = y/3.0 - 40
            if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                BB.append((x, y, w, h))

            (x, y) = elemento.pt
            h = 55
            w = 30
            x = x/3.0 - 41
            y = y/3.0 - 55
            if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                BB.append((x, y, w, h))

            (x, y) = elemento.pt
            h = 20
            w = 10
            x = x/3.0 - 5
            y = y/3.0 - 5
            if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
                BB.append((x, y, w, h))


    # BB = []
    # if len(kp) != 0:
    #     for elemento in kp:
    #
    #         (x, y) = elemento.pt
    #         cx = x/3.0
    #         cy = y/3.0
    #         diam = elemento.size/3.0
    #
    #         escalas = np.arange(0.5, 1.5, 0.5)
    #         diametros = diam * escalas
    #         a = np.arange(0.5, 2.0, 0.4)
    #         b = np.arange(4, 7)
    #
    #         W = [[i, j] for i in a for j in diametros]
    #         H = [[i, j] for i in b for j in diametros]
    #
    #         w = [i[0]*i[1] for i in W]
    #         h = [j[0]*j[1] for j in H]
    #
    #         x = cx - diametros
    #         y = cy - diametros
    #
    #         bb = [[round(x1), round(y1), round(w1), round(h1)] for x1 in x for y1 in y for w1 in w for h1 in h
    #         if (x1 + w1) < 164 and (y1 + h1) < 129 and x1 > 0 and y1 > 0]
    #         BB = BB + bb
            # if (x + w) < img.shape[1] and (y + h) < img.shape[0] and x > 0 and y > 0:
            #     BB.append((x, y, w, h))

    return BB


# img = cv2.imread('/home/luis/MEGA/Datasets/LSIFIR/Detection/Train/02/00450.png',0)
# d = proyeccion_vertical_gradiente(img)
#
# plt.figure(1)
# plt.imshow(d)
# plt.show()
# img1 = roi_umbral_dual_adaptativo(d, 'LSIFIR')
# plt.figure(2)
# plt.imshow(img1)
# plt.show()
