'''
Genera las anotaciones de la base de datos CV-09 y
guarda cada uno de esos archivos respecto a un tamaño:
h_base = 129.0
w_base = 165.0
'''
import math
import numpy as np
import os.path
##########################################

class Escribir_CV:

    def __init__(self, ruta_anotacion):
        self.ruta = ruta_anotacion
        self.target_folder_anotaciones = '/home/luis/MEGA/Datasets/CVCInfrared/'\
        'NightTime/Test/anotaciones/'
        self.ruta_base_folder_imagenes = '/home/luis/MEGA/Datasets/CVCInfrared/'\
        'NightTime/Test/FramesPos/'
        self.escribir_archivo()

    def escribir_archivo(self):

        nombre = self.ruta.split('/')[-1]
        nombre = nombre[0:len(nombre)-4]
        ruta_img = self.buscar_ruta_img()
        #xmin, ymin, xmax, ymax
        lista_bb = self.buscar_lista_gt()
        lista_bb = self.cambiar_escala(lista_bb)


        ruta_carpeta_anotaciones = self.target_folder_anotaciones
        print ruta_carpeta_anotaciones
        archivo = open(ruta_carpeta_anotaciones + nombre + '.txt', 'w')
        print ruta_img
        archivo.write(ruta_img + '\n')

        for elemento in lista_bb:
            print elemento
            pt_min = elemento[0:2]
            print pt_min
            pt_max = elemento[2:4]
            print pt_max
            archivo.write(str(pt_min) + ' - ' + str(pt_max) + '\n')

        archivo.close()


    def cambiar_escala(self, lista_bb):
        h_base = 129.0
        w_base = 165.0
        h_img = 480.0
        w_img = 640.0

        escala_w = w_base / w_img
        escala_h = h_base / h_img

        lista_bb_escalados = []
        for elemento in lista_bb:
            xmin = math.floor(elemento[0] * escala_w)
            xmax = math.floor(elemento[2] * escala_w)

            ymin = math.floor(elemento[1] * escala_h)
            ymax = math.floor(elemento[3] * escala_h)

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > w_base:
                xmax = w_base - 1
            if ymax > h_base:
                ymax = h_base - 1

            lista_bb_escalados.append(( int(xmin), int(ymin), int(xmax), int(ymax) ))

        return lista_bb_escalados


    def buscar_ruta_img(self):
        ruta_base = self.ruta_base_folder_imagenes
        nombre = self.ruta.split('/')[-1]
        nombre = nombre[0:len(nombre)-4]
        ruta_img = ruta_base + nombre + '.png'
        return ruta_img

    def buscar_lista_gt(self):
    	archivo = open(self.ruta, 'r')
    	aux = archivo.read().split()
        aux_int = [int(i) for i in aux]
        print len(aux_int)
    	lista_leida = [aux_int[i:i+4] for i in xrange(0, len(aux_int), 11)]
        print lista_leida
    	archivo.close()
        #lista_leida [xc, yc, w, h]
        lista_bb = []
        #lista_bb [x1,y1,x2,y2]
        for elemento in lista_leida:
            w = float(elemento[2]) - 1
            h = float(elemento[3]) - 1
            x1 = elemento[0] - w / 2.0
            y1 = elemento[1] - h / 2.0
            lista_bb.append([ x1, y1, x1+w, y1+h])

        print lista_bb
        return lista_bb
#####################################################################
# Ejecucion de la clase
# ruta_anot = '/home/luis/MEGA/Datasets/CVCInfrared/NightTime/Train/Annotations/000059.txt'
# obj = Escribir_CV(ruta_anot)
for i in range(1417, 8610):
    ruta_base = '/home/luis/MEGA/Datasets/CVCInfrared/NightTime/Test/Annotations/'
    num_img = "{:06d}".format(i)
    ruta_anot_1 = ruta_base + num_img + '.txt'
    ruta_anot_2 = ruta_base + num_img + '_2' '.txt'
    ruta_anot_3 = ruta_base + num_img + '_3' '.txt'
    if(os.path.isfile(ruta_anot_1)):
        obj = Escribir_CV(ruta_anot_1)

    if(os.path.isfile(ruta_anot_2)):
        obj = Escribir_CV(ruta_anot_2)

    if(os.path.isfile(ruta_anot_3)):
        obj = Escribir_CV(ruta_anot_3)
