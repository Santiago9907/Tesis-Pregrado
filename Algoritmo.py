import numpy as np
from numpy.linalg import norm
from numpy import round
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import Reticulo as rt

#Ajuste para gráfica recompensa
def mensaje(recompensa):
    if recompensa<0:
        print('------------------------------------------')
        print('------------------Choca!------------------')
        print('------------------------------------------')
    if recompensa>0:
        print('------------------------------------------')
        print('------------------Llegó!------------------')
        print('------------------------------------------')

#Política parametrizada
def politica(x,a,theta, espacio_acciones=['1','2','3','4']):
    numerador = np.exp((theta.T@rt.phi(x,a)).toarray())[0][0]
    iter = [np.exp((theta.T@rt.phi(x,accion)).toarray()) for accion in espacio_acciones]
    denominador = np.sum(iter)
    return numerador/denominador

#Toma de acción
def toma_de_accion(x,theta, espacio_acciones=['1','2','3','4']):
    probabilidades = [politica(x,a,theta,espacio_acciones) for a in espacio_acciones]
    accion_tomada = np.random.choice([1,2,3,4], replace=True, p=probabilidades)
    return int(accion_tomada),probabilidades

'''
Actualizaciones del Actor y del Crítico:
Los pasos serán alpha = 1/n y beta = 1/(1+n*ln(n))
'''

#Actualización del crítico
def critico(v, td_error, x, contador):
    paso = 1/contador
    v = v + paso*td_error*rt.f_x(x)
    return paso, v

#Actualización del actor
def actor(theta, td_error, x, a, contador):
    paso = 1/(1+np.log(contador)*contador)
    theta = theta + paso*td_error*rt.phi(x,a)
    theta[theta>10], theta[theta<-10] = 10,-10
    return paso, theta

def actor_critic(x_0, c, theta, v, dim=70, epsilon = 0.001):
    #Inicializo gráfico
    fig = plt.figure()
    rt.graficar_reticulo(fig)

    #Inicializo policy-parameter, value function, weight vector y la estimación de mi A.R.
    #v = csr_matrix(np.zeros((dim**2,1)).reshape((dim**2,1)),dtype=float)
    #theta = csr_matrix(np.zeros((4*(dim**2),1),dtype=float))
    J = 0
    termino = False
    contador = 0

    #Proceso iterativo
    while not termino:
        accion = toma_de_accion(x_0, theta)[0]
        #probabilidades = toma_de_accion(x_0, theta)[1]
        x_t = rt.decision(x_0, accion)
        recompensa = rt.recompensa(x_t)
        #mensaje(recompensa)
        contador += 1

        #Actualización del A.R.
        alpha = critico(v,0,x_0,contador)[0]
        J = (1-c*alpha)*J + (c*alpha)*recompensa

        #TD-error
        td_error = recompensa - J + (v.T@rt.f_x(x_t) - v.T@rt.f_x(x_0)).toarray()[0][0]

        #Reporte
        print('Recompensa: ' + str(recompensa) + '|| Td-error: ' + str(round(td_error,4)) + " || E: " + 
                str(x_0) + " || Theta: " + str(round(norm(theta),4)) + 
                " || v: " + str(round(norm(v),4)) + '|| n: ' + str(contador))
        plt.scatter(x_t[0], x_t[1], color = 'black')
        plt.pause(0.005)

        #Actualizaciones del Actor y del Crítico
        v = critico(v, td_error, x_t, contador)[1]
        theta_t = actor(theta, td_error, x_t, accion, contador)[1]
        
        #Verifica si termino
        if contador == 160000: #Cambiar condición
            termino = True
            print('Llegó')
        theta = theta_t
        x_0 = x_t
    return theta, v