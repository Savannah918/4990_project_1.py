from scipy import constants
from scipy.optimize import line_search
import numpy as np
NORM = np.linalg.norm
import numdifftools as nd
pi = np.pi

## we need to evaluate the 'conjugate' property

def func(x): # Objective function
    return 10*len(x) + sum(s**2-10*np.cos(2*pi*s) for s in x)                                 #Rastrigin
    return 0.5 + ((np.sin(x[0]**2-x[1]**2))**2-0.5) / (1+0.001*(x[0]**2+x[1]**2))**2             #Schaffer
    return 1 + 1/4000 * (x[0]**2+x[1]**2) - np.cos(x[0])*np.cos(x[1]/np.sqrt(2))                   #Griewank
    return (1 - x[0])**2 + 100*(x[0]**2 - x[1])**2 + (1 - x[1])**2 + 100*(x[1]**2 - x[2])**2         #Rosenbrock
# def grad_func(x): # Objective function
#     return numpy.array([4*x[0]**3-4*x[0]*x[1]+2*x[0]-2, -2*x[0]**2+2*x[1]])

def Hager_Zhang(Xj, tol, alpha_1, alpha_2):
    x = np.zeros((len(Xj), 1))
    for i in range(len(Xj)):
        x[i] = [Xj[i]]
    grad_func = nd.Gradient(func)
    D = grad_func(Xj)  #First Gradient
    delta = -D
    iter = 0
    while True:
        iter+=1
        start_point = Xj # Start point for step length selection 
        beta = line_search(f=func, myfprime=grad_func, xk=start_point, pk=delta, c1=alpha_1, c2=alpha_2)[0] # Selecting the step length
        if beta!=None:
            X = Xj + beta*delta
        
        if np.linalg.norm(grad_func(X)) < tol:
            for i in range(len(Xj)):
                x[i] += [Xj[i], ]
                print("iter:", iter)
            return X, func(X)
        else:
            Xj = X
            d = D
            D = grad_func(Xj)
            Q = D - d
            M = Q - 2*delta*NORM(Q)**2/(delta.dot(Q))
            N = D/(delta.dot(Q))
            chi = M.dot(N) # See line (19)
            temp_delta = delta
            delta = -D + chi*delta
            print(delta@temp_delta)
            for i in range(len(Xj)):
                x[i] += [Xj[i], ]

Initial_data = [5,5]
x,func_val = Hager_Zhang(Initial_data, 10**-6, 10**-4, 0.2)
print(x, func_val)