import math 
import numpy as np
import os
import win32com.client as win32
import time
import random

'''Case1'''

def Case1(x):
    def objective(x):
        x1 = x[0]
        x2 = x[1]
        return ((x1+10)/15 - 0.4)**2 + ((x2+15)/30 - 0.3)**2
    def constraint1(x):
        return -2*x[0] + x[1] - 15
    def constraint2(x):
        return (x[0]**2)/2 + 4*x[0] - x[1] - 5
    def constraint3(x):
        return -(x[0] - 4)**2 / 5 - 2*x[1]**2 + 10
    obj = objective(x)
    c1 = constraint1(x)
    c2 = constraint2(x)
    c3 = constraint3(x)

    return obj,c1,c2,c3

    
'''Case2'''

def Case2(x):
    def objective(x):
        x1 = x[0]
        x2 = x[1]
        return -(x1 - 10)**2 - (x2 - 15)**2
    def constraint(x):
        x1 = x[0]
        x2 = x[1]
        return ((x2 - 5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1 - 6)**2 + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 5)
    obj = objective(x)
    c1 = constraint(x)
    return obj,c1



'''Case3'''

def Case3(x):
    def Ackley(x):
        # xx = [x1, x2]
        a = 20
        b = 0.2
        c = 2 * math.pi
        sum1 = 0
        sum2 = 0
        for ii in range(1):
            xi = x[ii]
            sum1 = sum1 + xi**2
            sum2 = sum2 + math.cos(c*xi)
        term1 = -a * math.exp(-b * math.sqrt(sum1/2))
        term2 = -math.exp(sum2/2)
        y = term1 + term2 + a + math.exp(1)
        return y
    def branin_constraint(x):
        branin_val = (
        1 * (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0]**2 + (5 / np.pi) * x[0] - 6)**2 +
        10 * (1 - (1 / (8 * np.pi))) * np.cos(x[0]) + 10 +
        1 * (x[4] - (5.1 / (4 * np.pi ** 2)) * x[3]**2 + (5 / np.pi) * x[3] - 6)**2 +
        10 * (1 - (1 / (8 * np.pi))) * np.cos(x[3]) + 10 - 100)
        return branin_val
    obj = Ackley(x)
    c1 = branin_constraint(x)
    c2 = np.sum(np.sin(2 * np.pi * x)) - np.prod(np.cos(3 * np.pi * x))
    c3 = np.sum(np.exp(-x**2)) + np.prod(np.sin(x))
    c4 = 0.5 * np.sin(2 * np.pi * (x[0] - 2 * x[1])) + x[0] + 2 * x[1] - 1.5 + 3 * x[2] - 2 * x[3] +0.4*x[4]**2
    
    return obj, c1, c2, c3, c4
