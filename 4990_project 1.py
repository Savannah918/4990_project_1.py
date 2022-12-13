from scipy import constants
from scipy.optimize import line_search
import numpy
NORM = numpy.linalg.norm
import numdifftools as nd
import matplotlib.pyplot as plt

print('haha')
def func(x): # Objective function
    # return x[0]**4 - 2*x[0]**2*x[1] + x[0]**2 + x[1]**2 - 2*x[0] + 1
    return (1 - x[0])**2 + 100*(x[0]**2 - x[1])**2 + (1 - x[1])**2 + 100*(x[1]**2 - x[2])**2
# def grad_func(x): # Objective function
#     return numpy.array([4*x[0]**3-4*x[0]*x[1]+2*x[0]-2, -2*x[0]**2+2*x[1]])
# This is a test
true_sol = [1, 1, 1]

def Hager_Zhang(Xj, tol, alpha_1, alpha_2):
    x = numpy.zeros((len(Xj), 1))
    for i in range(len(Xj)):
        x[i] = [Xj[i]]
    grad_func = nd.Gradient(func)
    D = grad_func(Xj)
    delta = -D
    iterations = 0
    converge = []
    while True:
        iterations += 1
        converge.append(NORM(true_sol - Xj))
        start_point = Xj # Start point for step length selection 
        beta = line_search(f=func, myfprime=grad_func, xk=start_point, pk=delta, c1=alpha_1, c2=alpha_2)[0] # Selecting the step length
        if beta!=None:
            X = Xj + beta*delta

        
        if numpy.linalg.norm(grad_func(X)) < tol:
            for i in range(len(Xj)):
                x[i] += [Xj[i], ]

            print("iter:", iterations)
            plt.plot(range(iterations), converge)
            # plt.show()
            return X, func(X)
        else:
            Xj = X
            d = D
            D = grad_func(Xj)
            Q = D - d
            M = Q - 2*delta*NORM(Q)**2/(delta.dot(Q))
            N = D/(delta.dot(Q))
            chi = M.dot(N) # See line (19)
            delta = -D + chi*delta
            for i in range(len(Xj)):
                x[i] += [Xj[i], ]

initial = numpy.random.rand(3)
x, func_val = Hager_Zhang(initial, 10**-6, 10**-4, 0.2)
print(x, func_val)