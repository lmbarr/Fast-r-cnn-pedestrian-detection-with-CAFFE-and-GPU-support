#
'''
Escribe a continuacion del archivo, escribe la base de datos de CV, solo
los candidatos positivos.
Programa que genera un archivo txt con el siguiente formato.
# num_imagen
ruta_img
clase iou x1 y1 x2 y2
clase iou x1 y1 x2 y2
clase iou x1 y1 x2 y2
.............
.............
'''

import argparse
import time
import cv2
import numpy as np
import re
import os
import math
import random

def get_gts(ruta_anotacion):
	"""Devuelve una lista de todos los gt de una imagen
	lista = [gt1, gt2, gt3, ...]
	gtn es una tupla de 4 elementos	(x1,y1,x2,y2)"""
	# anotaciones = '/home/luis/MEGA/Datasets/LSIFIR/Detection/Train/annotations/'
	archivo_anotacion = open(ruta_anotacion, 'r')
	aux = re.findall('([0-9]+, [0-9]+)', archivo_anotacion.read())
	aux = str(aux)
	# Lista de string de los ncoordenada de los rois
	lista = re.findall('[0-9]+', aux)
	lista = [int(i) for i in lista]
	archivo_anotacion.close()
	return [lista[i:i+4] for i in xrange(0, len(lista), 4)]


def escribir_rois(file, ruta_anotacion, ruta_img):
	# Escribir los gts en el archivo
	lista_gts = get_gts(ruta_anotacion)

	print ruta_anotacion
	print ruta_img
	print lista_gts

	for gt in lista_gts:
		file.write('1'+' '+'1'+' '+str(gt[0])+' '+
		str(gt[1])+' '+str(gt[2])+' '+str(gt[3])+'\n')

#########################################

ruta_anotaciones = '/home/luis/MEGA/Datasets/CVCInfrared/NightTime/Train/anotaciones/anotaciones.txt'

archivo_train = open('/home/luis/MEGA/red_caffe/nueva_red_compleja/Train.txt','a')
j = 3148 # el numero que sigue del anterior conteno ver archiovo train
with open(ruta_anotaciones) as f:
	for line in f:
		ruta_anotacion = line[:-1]
		archivo_train.write('# ' + str(j) + '\n')

		with open(ruta_anotacion, 'r') as f:
			first_line = f.readline()[:-1]

		archivo_train.write(first_line + '\n')
		escribir_rois(archivo_train, ruta_anotacion, first_line)
		j = j + 1

archivo_train.close()
