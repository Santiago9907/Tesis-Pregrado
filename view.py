import numpy as np
import matplotlib.pyplot as plt
import Algoritmo as alg
from scipy.sparse import csr_matrix
import Reticulo as rt
import csv

#fig = plt.figure()
#rt.graficar_reticulo(fig)

#Estado inicial
dim = 70
x_0 = np.array([150,150])

#theta_0 = np.zeros((dim**2,1)).reshape((dim**2,1))
#v_0 = np.zeros((4*(dim**2),1)
with open('Resultados160k.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == "['theta'":
            theta_0 = np.array(list(map(float,row[1:-1] + [row[-1].replace(']','')]))).reshape((4*(dim**2),1))
        else:
            v_0 = np.array(list(map(float, row[1:-1] + [row[-1].replace(']','')]))).reshape((dim**2,1))

v_0 = csr_matrix(v_0, dtype=float)
theta_0 = csr_matrix(theta_0, dtype=float)

'''
Corro el algoritmo
'''

theta,v = alg.actor_critic(x_0, 0.95, theta_0, v_0)

'''
Guardo resultados
'''

rows = [theta,v]

np.savetxt("Resultados320k.csv", rows, delimiter =", ", fmt ='% s')







