'''
Genera el archivo Train.txt y Test.txt de la base de datos LSIFIR
necesario para el entrenamiento.
Programa que genera un archivo txt con el siguiente formato.
# num_imagen
ruta_img
clase iou x1 y1 x2 y2
clase iou x1 y1 x2 y2
clase iou x1 y1 x2 y2
.............
.............
'''
from helpers import IoU
import argparse
import time
import cv2
import numpy as np
import re
import os
from metodos_rois import detector_de_cabezas


def get_gts(ruta_img):
	'''
	Devuelve una lista de todos los gt de una imagen
	lista = [gt1, gt2, gt3, ...]
	gtn es una tupla de 4 elementos	(x1,y1,x2,y2).
	'''
	# anotaciones = '/home/luis/MEGA/Datasets/LSIFIR/Detection/Train/annotations/'
	aux = re.findall('[0-9]+', ruta_img)
	archivo = open(anotaciones+aux[0]+'_'+aux[1]+'.txt', 'r')
	aux = re.findall('([0-9]+, [0-9]+)', archivo.read())
	aux = str(aux)
	# Lista de string de los ncoordenada de los rois
	lista = re.findall('[0-9]+', aux)
	lista = [int(i) for i in lista]
	archivo.close()
	return [lista[i:i+4] for i in xrange(0, len(lista), 4)]


def escribir_rois(file, ruta_img):
	image = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
	# Escribir los gts en el archivo
	lista_gts = get_gts(ruta_img)
	for gt in lista_gts:
		file.write('1'+' '+'1'+' '+str(gt[0])+' '+
		str(gt[1])+' '+str(gt[2])+' '+str(gt[3])+'\n')

	# Retorna boxes en formato (x,y,w,h)
	boxes = detector_de_cabezas(image)

	for elemento in boxes:
		# calcular iou
		(x,y,w,h) = elemento
		lista_iou = []

		for gt in lista_gts:
			lista_iou.append(IoU((x,y,x+w,y+h), gt))

		iou = max(lista_iou)

		if iou > 0.55:
			clase = '1'
			file.write(clase+' '+str(iou)+' '+str(x)+' '+
			str(y)+' '+str(x + w)+' '+str(y + h)+'\n')

		# Solo agregar rois con iou mayor a 0.1
		elif iou >= 0.18 and iou <= 0.55:
			clase = '0'
			file.write(clase+' '+str(iou)+' '+str(x)+' '+
			str(y)+' '+str(x + w)+' '+str(y + h)+'\n')

#############################################################
import math
import random
conjunto = 'Train'

# crea el archivo
archivo_generado = open(conjunto + ".txt", "w")

archivo = open("/home/luis/MEGA/Datasets/LSIFIR/Detection/"+conjunto+"/pos.lst", 'r')

with archivo as f:
    total_img_pos = sum(1 for _ in f)

num_images = 3000
lista_imagenes = random.sample(range(total_img_pos), num_images)

j = 0
ruta_base = '/home/luis/MEGA/Datasets/LSIFIR/Detection/'
anotaciones = '/home/luis/MEGA/Datasets/LSIFIR/Detection/'+conjunto+'/annotations/'

for ele in lista_imagenes:
	with open('/home/luis/MEGA/Datasets/LSIFIR/Detection/'+conjunto+'/pos.lst') as fp:
	    for i, line in enumerate(fp):
	        if i == ele:
				ruta_img = ruta_base+line[:-1]
				aux = re.findall('[0-9]+', ruta_img)

				if os.path.isfile(anotaciones+aux[0]+'_'+aux[1]+'.txt'):
					archivo_generado.write('# '+str(j)+'\n')
					j = j + 1
					archivo_generado.write(ruta_base + line)
					# line incluye el salto de linea
					escribir_rois(archivo_generado, ruta_img)
				else:
					pass

archivo.close()
archivo_generado.close()
