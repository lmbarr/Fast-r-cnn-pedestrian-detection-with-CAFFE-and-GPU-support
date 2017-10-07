'''
Guarda la salida de la red de toda la base de datos LSIFIR Test
para despues generar graficas del detector.
'''



from Detector import Detector
import sys
import os
import cv2
import gc
import numpy as np
from fppi import Metricas_Detector


detector = Detector()
ruta_img = '/home/luis/MEGA/Datasets/LSIFIR/Detection/Test/'
umbrales = [0.1]
lista_miss_rate = []
lista_fppi = []
lista_dt = []

for k in umbrales:
    metrica = Metricas_Detector(conjunto='Test')
    for j in range(1, 8):
        for i in range(3800):

            idx = "{:05d}".format(i)
            escena = "{:02d}".format(j)
            ruta = ruta_img + escena + '/' + idx + '.png'

            if not os.path.isfile(ruta):
                continue

            print ruta

            img = cv2.imread(ruta, 0)
            detector.empezar_deteccion(img, k)
            # Guardar la salida de la red con la ruta de la imagen
            carpeta_target = '/home/luis/Desktop/EntregablesTesis/ultima_red/GraficasDetector/salida_detector/'
            nombre = escena + '_' + idx + '.npz'

            if detector.get_salida_red_cruda() != 'None':
                np.savez(carpeta_target + nombre, name1=detector.get_salida_red_cruda()
                , name2=detector.get_BB_crudo(), name3=ruta)
            # data = np.load('/home/luis/Desktop/mat.npz')
            ###################################################

            # BB_final = detector.get_BB_final()
            # scores = detector.get_pro()
            # metrica.determinar_valores1(ruta, BB_final, scores)
            # metrica.incrementar_img()

#     metrica.calcular_fppi()
#     lista_fppi.append(metrica.get_fppi())
#     metrica.calcular_miss_rate()
#     lista_miss_rate.append(metrica.get_miss_rate())
#     metrica.calcular_tasa_deteccion()
#     lista_dt.append(metrica.get_tasa_deteccion())
#     print "parametros ", metrica.get_parametros()
#     print "fppi ", metrica.get_fppi()
#     print "num img ", metrica.get_num_img()
#     print "num gts ", metrica.get_contador_gts()
#     print "num de candidatos ", metrica.get_num_total_candidatos()
#     print "miss rate ", metrica.get_miss_rate()
#     print "DR ", metrica.get_tasa_deteccion()
#
#
#
# # saving
# ruta = '/home/luis/MEGA/red_caffe/ultima_red/ProgramaTesis/archivosotros/'
# f = open(ruta + "NMS_fppi_vs_miss_rate_0.9999", "w")
# f.write("# x y\n")        # column names
# np.savetxt(f, np.array([lista_fppi, lista_miss_rate]).T)
#
# # saving:
# f1 = open(ruta + "NMS_fppi_vs_DT_0.9999", "w")
# f1.write("# x y\n")        # column names
# np.savetxt(f1, np.array([lista_fppi, lista_dt]).T)
