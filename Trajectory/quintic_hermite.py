import math
import numpy as np
from matplotlib import pyplot as plt
import roboticstoolbox as rtb
import numpy as np
import math
from math import pi
from roboticstoolbox import tools as tools
import spatialmath as sm
import spatialgeometry as sg
import matplotlib as mpl
import matplotlib.pyplot as plt
from call_tau import *

def hermite_basis(t):
    # This function generates half of the Hermite base matrix
    HER_half = np.zeros((3, 6))
    # Base function
    HER_half[0, 0] = -6*t**5 + 15*t**4 - 10*t**3 + 1
    HER_half[0, 1] = -3*t**5 + 8*t**4 - 6*t**3 + t
    HER_half[0, 2] = -0.5*t**5 + 1.5*t**4 -1.5*t**3 +0.5*t**2
    HER_half[0, 3] = 6*t**5 - 15*t**4 + 10*t**3
    HER_half[0, 4] = -3*t**5 + 7*t**4 -4*t**3
    HER_half[0, 5] = 0.5*t**5 - t**4 +0.5*t**3       
    # First derivative of the base function
    HER_half[1, 0] = -30*t**4 + 60*t**3 - 30*t**2
    HER_half[1, 1] = -15*t**4 + 32*t**3 - 18*t**2 + 1
    HER_half[1, 2] = -2.5*t**4 + 6*t**3 - 4.5*t**2 + t
    HER_half[1, 3] = 30*t**4 - 60*t**3 + 30*t**2
    HER_half[1, 4] = -15*t**4 + 28*t**3 - 12*t**2
    HER_half[1, 5] = 2.5*t**4 - 4*t**3 + 1.5*t**2
    # Second derivative of the base function
    HER_half[2, 0] = -120*t**3 + 180*t**2 -60*t
    HER_half[2, 1] = -60*t**3 + 96*t**2 -36*t
    HER_half[2, 2] = -10*t**3 + 18*t**2 - 9*t + 1
    HER_half[2, 3] = 120*t**3 - 180*t**2 + 60*t
    HER_half[2, 4] = -60*t**3 + 84*t**2 - 24*t
    HER_half[2, 5] = 10*t**3 - 12*t**2 + 3*t

    return HER_half

def solver(t0, t1, x_point, y_point, num): # x_point: [q, qd, qdd] at x, y_point: [q, qd, qdd] at y
    # Solves for the coefficients of quintic Hermite spline
    HER_upper = hermite_basis(t0)
    HER_lower = hermite_basis(t1)
    HER_total = np.concatenate((HER_upper, HER_lower), axis = 0)
    xval = np.transpose(x_point)
    yval = np.transpose(y_point)
    val_total = np.concatenate((xval, yval), axis = 0)
    coeff = np.linalg.solve(HER_total, val_total)
    func = np.zeros(num+1)
    func_d = np.zeros(num+1)
    func_dd = np.zeros(num+1)
    time_spline = np.zeros(num+1)
    delta = (t1-t0)/num
    i = 0
    n = t0
    #print(n)
    #print(t1)
    #print(f't0 = {t0}, loop ends when n >= {np.round(t1, 4)}')

    # Evaluate each point between t0 and t1 using quintic hermite base functions, results stored in func[] as a stacked matrix
    # Time steps stored in time_spline[] as a stacked matrix
    while n < t1 + delta/100: # plus a small margin to compensate floating point error
        func[i] = hermite_basis(n)[0, :].dot(coeff)
        func_d[i] = hermite_basis(n)[1, :].dot(coeff)
        func_dd[i] = hermite_basis(n)[2, :].dot(coeff)
        time_spline[i] = n
        #print(f'array func at index {i} has the value of {func[i]}, n = {n}')
        i = i + 1
        #print(i)
        n = n + delta
        #print(n)
    #print(f'loop breaks at i = {i}, n = {n}')
    #print(f'what is func? {func}')
    return func, time_spline, func_d, func_dd

def quintic_hermite_approx(num, matrix_eval, time_vec, step_n): 
    # num: number of points for trajectory generation, 
    # joint_n: joint number(0-5)
    # step_n: number of points for each hermite interpolation
    flat_func_reconstr = np.zeros((num-1)*step_n)

    func = list()
    time = list()
    func_d = list()
    func_dd = list()
    """
    func1 = np.zeros(step_n+1)
    time1 = np.zeros(step_n+1)
    func2 = np.zeros(step_n)
    time2 = np.zeros(step_n)
    """
    time_spline = np.zeros((num, step_n+1))

    for i in range(num-1):
        x_point = matrix_eval[i, :]
        y_point = matrix_eval[i+1, :]
        """
        if i == 0:
            func1 = solver(time_vec[i], time_vec[i+1], x_point, y_point, step_n)[0]
            time1 = solver(time_vec[i], time_vec[i+1], x_point, y_point, step_n)[1]
            func = func1
            time = time1
            #print(f'what is func1{func1}')
        else:
            func_org = solver(time_vec[i], time_vec[i+1], x_point, y_point, step_n)[0]
            time_org = solver(time_vec[i], time_vec[i+1], x_point, y_point, step_n)[1]
            func2 = func_org[1:]
            time2 = time_org[1:]
            #print(step_n)
            #print(f'original func{func_org}')
            #print(f'what is func2? {func2}')
            func = np.concatenate((func, func2))
            time = np.concatenate((time, time2))
        """

        func.append(solver(time_vec[i], time_vec[i+1], x_point, y_point, step_n)[0])
        time.append(solver(time_vec[i], time_vec[i+1], x_point, y_point, step_n)[1])
        func_d.append(solver(time_vec[i], time_vec[i+1], x_point, y_point, step_n)[2])
        func_dd.append(solver(time_vec[i], time_vec[i+1], x_point, y_point, step_n)[3])
    #flat_func = func.flatten()

    return func, time, func_d, func_dd
    #flat_time_spline = time_spline.flatten()
    #print(func.flatten())
    #print(time_spline.flatten())


"""
    func = np.zeros((20, 10+1))
    time_spline = np.zeros((20, 10+1))

    for i in range(20):
        x_point = np.array([rdr[3].q[i], rdr[3].qd[i], rdr[3].qdd[i]])
        y_point = np.array([rdr[3].q[i+1], rdr[3].qd[i+1], rdr[3].qdd[i+1]])
        func[i, :] = solver(rdr[6][i], rdr[6][i+1], x_point, y_point, 10)[0]
        time_spline[i, :] = solver(rdr[6][i], rdr[6][i+1], x_point, y_point, 10)[1]

    flat_func = func.flatten()
    flat_time_spline = time_spline.flatten()
    print(func.flatten())
    print(time_spline.flatten())

    plt.plot(flat_time_spline, flat_func)
    plt.plot(rdr[6], np.round(rdr[3].q, 4))
    plt.show()
"""

# To do : integrate the results from HKA into the quintic_hermite_approx to generate optimized trajectory