import numpy as np
import matplotlib.pyplot as plt
import Algoritmo as alg
from scipy.sparse import csr_matrix
import Reticulo as rt

#fig = plt.figure()
#rt.graficar_reticulo(fig)

#Estado inicial
dim = 70
x_0 = np.array([160,150])
v_0 = csr_matrix(np.zeros((dim**2,1)).reshape((dim**2,1)),dtype=float)
theta_0 = csr_matrix(np.zeros((4*(dim**2),1),dtype=float))

theta,v = alg.actor_critic(x_0, 0.95, theta_0, v_0)


