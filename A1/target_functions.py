import numpy as np

def Rastrigin(x):
    """Rastrigin Function, Highly multi-peak"""
    x = np.asarray(x)
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def Sphere(x):
    """Sphere Function, unimodal function"""
    return np.sum(x**2)

def Griewank(x):
    n = len(x)
    sum1 = np.sum(np.square(x)) / 4000
    prod2 = np.prod([np.cos(x[i] / np.sqrt(i+1)) for i in range(n)])
    return sum1 - prod2 + 1

def Rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)