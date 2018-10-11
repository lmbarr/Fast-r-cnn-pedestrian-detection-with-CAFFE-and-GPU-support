'''Archivo principal que se ejecuta para comenzar el entrenamiento
una vez configurado los archivos:
* solver.prototxt
* capa_datos.py
* train_test.prototxt
* Train.txt
* Test.txt'''

import sys
sys.path.insert(0, "/home/luis/fast-rcnn/caffe-fast-rcnn/python")
import caffe
################################################
ruta = "/home/luis/Desktop/EntregablesTesis/Entrenamiento/"
solver = caffe.get_solver(ruta + "solver.prototxt")
# solver.restore(ruta + 'pesos2/_iter_90000.solverstate');
solver.solve()
