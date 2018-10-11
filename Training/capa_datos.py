''' Capa de caffe tipo python que lee los datos del
archivo Train.txt y los alimenta a la entrada de la
arquitectura para comenzar el entrenamiento'''

import sys
sys.path.insert(0, "/home/luis/fast-rcnn/caffe-fast-rcnn/python")
import caffe
import numpy as np
from random import sample
import re
import math
from random import sample
from helpers import isint, isfloat, gradiente
import yaml
import cv2

class CapaDatos(caffe.Layer):

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.conjunto = layer_params['conjunto']

        ruta_train = '/home/luis/MEGA/red_caffe/ultima_red/'+self.conjunto+'.txt'
        num = 5346 if 'Train' == self.conjunto else 2980

        self.num_imagenes_por_batch = 10
        self.num_total_imgs = num
        self.lista = range(self.num_total_imgs)

        archivo = open(ruta_train, 'r')

        self.archivo_train = archivo.read()
        archivo.close()

        top[0].reshape(1, 2, int(math.ceil(129*1.4)), int(165*1.4))
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[1].reshape(1, 5)
        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[2].reshape(1)
        h_base = 129.0*1.4
        w_base = 165.0*1.4
        h_img = 129.0
        w_img = 165.0
        self.escala_w = w_base / w_img
        self.escala_h = h_base / h_img

    def seleccionar_rois(self, archivo_train, lista):
        """
        Del numero de imagenes escogidas al azar dado por num_imagenes_por_batch
        selecciona al azar rois negativos y positivos para cada fotograma
        seleccionado.

        Args:
            archivo_train (str): Archivo txt donde se busca
            el fotograma y selecciona los rois.

            lista (list): Lista de imagenes escogidas de la base de datos.

        Returns:
            lista_rois (list): Lista de rois, donde cada elemento tiene
            la forma [n, x1, y1, x2, y2] donde n indica a que imagen pertenece
            ese roi.
        """
        lista_rois = []
        labels = []
        # numero de rois negativos por fotograma
        num_rois_neg = 10
        # Busca entre las imagenes seleccionadas
        for i in lista:
            f = re.findall('# '+str(i)+'\n[a-zA-Z0-9./_]+[0-9. \n]+', archivo_train)
            b = re.findall('\n[0-9 .]+', f[0])

            # Escoger 14 rois por imagen(4 pos y 10 sorteados)
            # Generalmente los primeros candidatos de la lista de rois
            # son positivos
            try:
                ind_rois = sample(range(1,len(b),1), num_rois_neg)
                ind_rois.append(0)
                ind_rois.append(1)
                ind_rois.append(2)
                ind_rois.append(4)
            except ValueError:
                ind_rois = range(len(b))


            for k in ind_rois:
                aux = b[k][1:].split()
                # print aux
                aux = [float(j) for j in aux]
                # Guardar el primer termino (la clase)
                clase = int(aux[0])

                # Asignar al primer termino el ind_roi
                aux[0] = lista.index(i)
                # Eliminar el iou de cada lista
                aux.pop(1)
                # Ajuste de escala
                if len(aux) == 5:
                    aux[1] = aux[1] * self.escala_w
                    aux[2] = aux[2] * self.escala_h
                    aux[3] = aux[3] * self.escala_w
                    aux[4] = aux[4] * self.escala_h
                    lista_rois.append(aux)
                    labels.append(clase)

        # print 'funcion', len(lista_rois)
        # print lista_rois

        return np.array(lista_rois, dtype=np.int16), np.array(labels, dtype=np.int8)


    def obtener_imagenes(self, lista_rutas):
        """
        Con la lista de rutas de las imagenes, obtiene las imagenes
        y las procesa de tal manera que queden listas para entrar en
        la arquitecutra.

        Args:
            lista_rutas (list): Lista de las ruta de las imagenes escogidas.

        Returns:
            numpy.ndarray: Numpy array que entra a la arquitecura.
        """
        # print lista_rutas np.float32(img) / 255.0  ,
        datos_img = map(lambda x: cv2.resize(np.float32(cv2.imread(x, 0)) / 255.0, (int(165*1.4), int(math.ceil(129*1.4)) )), lista_rutas)
        datos_img = map(lambda fig: ((1 - 0) /( np.amax(fig) - np.amin(fig)) ) * ( fig - np.amax(fig) ) + 1, datos_img)
        datos_grad = map(lambda y: gradiente(y), datos_img)
        lista_aux = []
        lista_definitiva = []

        for i in range(len(lista_rutas)):
            lista_aux.append(datos_img[i])
            lista_aux.append(datos_grad[i])
            lista_definitiva.append(np.asarray(lista_aux))
            lista_aux = []


        return np.asarray(lista_definitiva)

    def ruta_imagenes(self, archivo_train, lista):

        '''
        De la lista de imagenes escogidas, busca las imagenes segun
        el numero en Train.txt y retorna una lista
        de las rutas de las imagenes.

        Args:
            lista (list): Lista numerica de imagenes escogidas.

        Returns:
            lista_rutas: lista de rutas.
        '''

        lista_rutas = []
        for i in lista:
            informacion_img = re.findall('# '+str(i)+'\n[a-zA-Z0-9./_]+', archivo_train)
            g = re.findall('/[a-zA-Z0-9/._]+', informacion_img[0])
            lista_rutas.append(g[0])

        return lista_rutas


    def get_next_minibatch(self):
        '''
        Accede a los datos de que sirven para adquirir lista
        de rois, lista de grounth truth (labels) y lista de imagenes.

        Returns:
            imagenes (numpy.ndarray): lista de imagenes de 2 canales.
            lista_rois (numpy.ndarray): lista de rois.
            labels (numpy.ndarray): lista de ground truth.
        '''
        ind_imagenes = sample(self.lista, self.num_imagenes_por_batch)

        lista_rois, labels = self.seleccionar_rois(self.archivo_train, ind_imagenes)
        lista_rutas = self.ruta_imagenes(self.archivo_train, ind_imagenes)
        imagenes = self.obtener_imagenes(lista_rutas)

        return imagenes, lista_rois, labels



    def forward(self, bottom, top):
        '''Carga los datos a la red para cada epoca.'''
        [imagenes, rois, labels] = self.get_next_minibatch()



        top[0].reshape(*(imagenes.shape))
        # Copy data into net's input blobs
        top[0].data[...] = imagenes.astype(np.float32, copy=False)

        top[1].reshape(*(rois.shape))
        # Copy data into net's input blobs rois.astype(np.float32, copy=False)
        # print type(rois)
        # print rois.shape
        # print rois
        top[1].data[...] = rois

        top[2].reshape(*(labels.shape))
        # Copy data into net's input blobs
        top[2].data[...] = labels


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
