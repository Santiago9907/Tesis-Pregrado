#Política exacta: L.P.
import numpy as np
from numpy import zeros
from numpy.random import choice
import scipy as sp
from scipy.sparse import csr_matrix, vstack
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Constantes del problema
'''
Zonas prohibidas y zona objetivo
'''
zona_objetivo = [155,156,157]
region1 = {'x':[15,16],'y':np.arange(start=40, stop=100)}
region2 = {'x':(np.arange(start=40, stop=90)).tolist(),'y':[60,61,62,63,64,65,66]}
region3 = {'x':(np.arange(start=100, stop=140)).tolist(),'y':[140,141,142,143]}
region4 = {'x':(np.arange(start=10, stop=30)).tolist(),'y':[180,181,182,183,184,185]}
region5 = {'x':(np.arange(start=105, stop=125)).tolist(),'y':[90,91,92,93,94]}
region6 = {'x':[169,170,171,172,173],'y':(np.arange(start=130, stop=150)).tolist()}
region7 = {'x':[50,51,52,53,54,55,56],'y':(np.arange(start=120, stop=170)).tolist()}
region8 = {'x':(np.arange(start=140, stop=170)).tolist(),'y':[25,26,27]}
zona_prohibida = [region1,region2,region3,region4,region5,region6,region7]
borde = [1,199]

def graficar_reticulo(fig):
    ax = fig.add_subplot(111)
    ax.set_facecolor('#EBEBEB')
    r1 = plt.Rectangle((15, 40),2, 60, color = 'darkred')
    r2 = plt.Rectangle((40, 60),50, 7, color = 'darkred')
    r3 = plt.Rectangle((100, 140),40, 4, color = 'darkred')
    r4 = plt.Rectangle((10, 180),20, 6, color = 'darkred')
    r5 = plt.Rectangle((105,90),20, 5, color = 'darkred')
    r6 = plt.Rectangle((169,130),5, 20, color = 'darkred')
    r7 = plt.Rectangle((50,120),7, 50, color = 'darkred')
    r8 = plt.Rectangle((140,25),30, 3, color = 'darkred')
    r9 = plt.Rectangle((0,0),1, 200, color = 'black')
    r10 = plt.Rectangle((0,0),200, 1, color = 'black')
    r11 = plt.Rectangle((0,199),200, 1, color = 'black')
    r12 = plt.Rectangle((199,0),1, 200, color = 'black')
    recompensa = plt.Rectangle((155,155), 2, 2, color = 'blue')
    rects = [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,recompensa]
    for rect in rects:
        ax.add_patch(rect)
    plt.xlim([0,200])
    plt.ylim([0,200])
    plt.title('Retículo 200x200 con obstáculos', fontdict={'family':'serif','color':'black','size':20})
    #plt.show()
    return ax
"""
Probabilidades
Subir = 1
Bajar = 2
Izquierda = 3
Derecha = 4
"""

acciones_reticulo = {'1':[0.3,0.4,0.2,0.1],'2':[0.3,0.3,0.2,0.2],
                    '3':[0.3,0.2,0.3,0.2],
                    '4':[0.3,0.4,0.2,0.1]}

arriba = {'a':[0,1], 'b':[0,2], 'c':[-1,2], 'd':[-1,1]}
abajo = {'a':[0,0], 'b':[0,-1], 'c':[-1,0], 'd':[-1,-1]}
izquierda = {'a':[-1,1], 'b':[-1,0], 'c':[-2,0], 'd':[-2,1]}
derecha = {'a':[1,0], 'b':[1,1], 'c':[0,0], 'd':[0,1]}

movs = ['a','b','c','d']

#Tile-Coding
dim_regillas = 40
num_regillas = 7
cant_regillas = 70
corrimientos = choice(np.arange(start=1, stop=dim_regillas), size=num_regillas, replace=False)


def decision(x_actual, accion_tomada, acciones=acciones_reticulo, d=200):
    """
    x_actual es vector en N^2 con las coordenadas de la posición actual
    acciones es un diccionario con las acciones como llaves y las probabilidades como valores
    """
    '''if x_actual[0] in zona_objetivo and x_actual[1] in zona_objetivo:
        x_actual[0] = choice(np.arange(0,200),replace=True)
        x_actual[1] = choice(np.arange(0,200),replace=True)'''
    
    accion_tomada = str(accion_tomada)
    if accion_tomada == '1':
        movimiento_realizado = arriba[choice(movs, size=1, replace=True, p=acciones['1'])[0]]
    if accion_tomada == '2': 
        movimiento_realizado = abajo[choice(movs, size=1, replace=True, p=acciones['2'])[0]]
    if accion_tomada == '3':
        movimiento_realizado = izquierda[choice(movs, size=1, replace=True, p=acciones['3'])[0]]
    if accion_tomada == '4':
        movimiento_realizado = derecha[choice(movs, size=1, replace=True, p=acciones['4'])[0]]

    #Se mueve el dron de acuerdo a la acción tomada    
    x_actual = x_actual + movimiento_realizado
    if x_actual[0]>=d or x_actual[0]<=1 or x_actual[1]<=1 or x_actual[1]>=d: 
        x_actual[0] = choice(np.arange(0,200),replace=True)
        x_actual[1] = choice(np.arange(0,200),replace=True)
    return x_actual

'''
Recompensas: Consideramos los dos casos de recompensas. El caso de -1,0,1 y el de 1 sobre la distancia
a la que se encuentra de la zona objetivo
'''

def recompensa(x_actual, tipo = 1):
    #recompensa -1,0,1
    if tipo == 1:
        for region in zona_prohibida:
            if x_actual[0] in region['x'] and x_actual[1] in region['y']: return -1000
        if x_actual[0] in borde or x_actual[1] in borde: return -100
        if x_actual[0] in zona_objetivo and x_actual[1] in zona_objetivo: return 1000
        else: return 0

    #recompensa según qué tan lejos está
    if tipo == 2:
        if x_actual[0] in zona_objetivo and x_actual[1] in zona_objetivo: return 10
        elif x_actual[0] in zona_prohibida or x_actual[1] in zona_prohibida: return -10
        else:
            diferencia = np.array([156,156]) - np.array(x_actual)
            return 1/np.linalg.norm(diferencia)

#TILE CODING

'''
Vectores phi:
Caso sin Tile-Coding: Con 40.000 estados y 4 acciones = 40.000*4 = 160.000 
vectores en R^160.000
Con Tile-Coding: Con 40.000 estados, se usarán 10 regillas de 70x70 secciones con 
corrimientos aleatorios donde cada una contendrá 50x50 estados generando vectores de 
49.000x4 por cada una de las acciones.
'''

def phi(x, accion, d=200, Tile = True, dim_regillas = dim_regillas, cant_regillas = cant_regillas, num_regillas = num_regillas):
    """
    dim_regillas: cantidad de estados que hay en cada regilla (dim_regillas x dim_regillas)
    cant_regillas: Cantidad de cuadrículas que va a tener la regilla (cant_regillas x cant_regillas)
    num_regillas: Cantidad de veces que se va a mover la regilla
    """
    if Tile == False:
        """
        x es vector en R^2 con las coordenadas de la posición actual
        """
        #vector de d^2 lleno de 0's donde en la posición del dron (x[0],x[1]) es un 1
        matriz = csr_matrix(arg1=([1], ([x[0]],[x[1]])), shape=[d,d]).reshape((d**2,1))
        #Vector de d^2 lleno de 0's
        vector_ceros = csr_matrix(np.zeros((d**2,1),dtype=float))

        #Retorna el vector acorde con la acción que se realizó
        if str(accion) == '1': 
            return vstack((matriz,vector_ceros,vector_ceros,vector_ceros))
        elif str(accion) == '2': 
            return vstack((vector_ceros,matriz,vector_ceros,vector_ceros))
        elif str(accion) == '3': 
            return vstack((vector_ceros,vector_ceros,matriz,vector_ceros))
        else: 
            return vstack((vector_ceros,vector_ceros,vector_ceros,matriz))
    '''
    Para el caso del Tile-Coding, asumiremos que el espacio en el que se va a mover el Dron es cuadrado
    '''
    if Tile == True:
        '''
        Para la regilla inicial, sobrarán la misma cantidad de espacios arriba y abajo. En el 
        caso de 200x200 y regilla 70x70, solo 50x50 cuadros cubrirán el espacio. Sobrarán 10
        arriba, 10 abajo, 10 a la izq y 10 a la derecha.
        '''
        #Ubicación de x en coordenadas de la regilla
        x_regilla = [x[0]%dim_regillas + 1, x[1]%dim_regillas + 1]
        phi = csr_matrix(arg1=([1], ([x_regilla[0]],[x_regilla[1]])), shape=[cant_regillas,cant_regillas]).reshape((cant_regillas**2,1))

        #Ubicación de x en coordenadas de las regillas nuevas que serán movidas aleatoriamente
        for i in range(num_regillas):
            #Mover la regilla (x,y) es mover el retículo (-x,-y)
            x_regilla = [(x[0]+corrimientos[i])%dim_regillas+1, (x[1]+corrimientos[i])%dim_regillas+1]
            phi += csr_matrix(arg1=([1], ([x_regilla[0]],[x_regilla[1]])), 
                                            shape=[cant_regillas,cant_regillas]).reshape((cant_regillas**2,1))

        #Distinción por acciones caso tile coding
        vector_ceros = csr_matrix(np.zeros((cant_regillas**2,1),dtype=float))
        #Retorna el vector acorde con la acción que se realizó
        if str(accion) == '1': 
            return vstack((phi,vector_ceros,vector_ceros,vector_ceros))
        elif str(accion) == '2': 
            return vstack((vector_ceros,phi,vector_ceros,vector_ceros))
        elif str(accion) == '3': 
            return vstack((vector_ceros,vector_ceros,phi,vector_ceros))
        else: 
            return vstack((vector_ceros,vector_ceros,vector_ceros,phi))
        

'''
Vectores f:
Caso sin Tile-Coding: a diferencia de phi, f solo depende del estado. Luego, f_x 
será un vector en R^40.000 
C
'''

def f_x(x, d=200, Tile = True, dim_regillas = dim_regillas, cant_regillas = cant_regillas, num_regillas = num_regillas):
    if Tile == False:
        return csr_matrix(arg1=([1], ([x[0]],[x[1]])), shape=[d,d]).reshape((d**2,1))
    else:
        x_regilla = [x[0]%dim_regillas + 1, x[1]%dim_regillas + 1]
        f = csr_matrix(arg1=([1], ([x_regilla[0]],[x_regilla[1]])), shape=[cant_regillas,cant_regillas]).reshape((cant_regillas**2,1))

        #Ubicación de x en coordenadas de las regillas nuevas que serán movidas aleatoriamente
        for i in range(num_regillas):
            #Mover la regilla (x,y) es mover el retículo (-x,-y)
            x_regilla = [(x[0]+corrimientos[i])%dim_regillas, (x[1]+corrimientos[i])%dim_regillas]
            f += csr_matrix(arg1=([1], ([x_regilla[0]],[x_regilla[1]])), 
                                            shape=[cant_regillas,cant_regillas]).reshape((cant_regillas**2,1))
        return f
        