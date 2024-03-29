import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def lerp_func(x1, x2, y1, y2, x):
    k = (y2 - y1)/(x2 - x1)
    #print(k)
    return k * (x - x1) + y1

def lerp_func_integral(x1, x2, y1, y2, x, qd0):
    k = (y2 - y1)/(x2 - x1)
    return (0.5 * k * x**2 - k * x1 * x + y1 * x - (0.5 * k * x1**2 - k * x1 * x1 + y1 * x1)) + qd0

def lerp_func_double_integral(x1, x2, y1, y2, x, qd0, q0):
    k = (y2 - y1)/(x2 - x1)
    a2 = qd0 - 0.5 * k * x1**2 + k * x1 * x1 - y1 * x1
    return 1/6 * k * x**3 - 0.5 * k * x1 * x**2 + 0.5 * y1 * x**2 + a2 * x - (1/6 * k * x1**3 - 0.5 * k * x1 * x1**2 + 0.5 * y1 * x1**2 + a2 * x1) + q0
