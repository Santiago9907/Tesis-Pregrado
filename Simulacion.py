import numpy as np
import matplotlib.pyplot as plt

x = 1
y = 1

def h(x,y):
    return 1/(np.sqrt(5 - x**2 + y**2))

def g(x,y):
    return 1/(- np.log(x) + np.log(y))

def alpha_n(n):
    return 1/(n+1)

def beta_n(n):
    return 1/(1+n*np.log(n))

def martingala():
    return 0

def recursion(x,y):
    #fig = plt.figure()
    for n in range(2000):
        x_n = x + alpha_n(n)*(h(x,y) + martingala())
        y_n = y + beta_n(n)*(g(x,y) + martingala())
        '''plt.scatter(n, x_n, color = 'black')
        plt.scatter(n, y_n, color = 'blue')
        plt.pause(0.005)'''
        x,y = x_n,y_n
        print('x: ' + str(x) + ' ||y: ' + str(y))

recursion(1,1)