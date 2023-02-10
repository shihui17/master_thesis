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

def generate_traj(num):
    Yu = rtb.models.DH.Yu()
    q_end = Yu.qa
    t = np.linspace(0, 5, num+1)
    tg1 = tools.trapezoidal(pi/2, q_end[0], num+1)
    tg2 = tools.trapezoidal(pi/3, q_end[1], num+1)
    tg3 = tools.trapezoidal(pi/4, q_end[2], num+1)
    tg4 = tools.trapezoidal(pi/6, q_end[3], num+1)
    tg5 = tools.trapezoidal(pi/3, q_end[4], num+1)
    tg6 = tools.trapezoidal(pi/6, q_end[5], num+1)
    return tg1, tg2, tg3, tg4, tg5, tg6, t

rdr = generate_traj(20)

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
    HER_half[1, 5] = 2.5*t**4 - 4**t*3 + 1.5*t**2
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
    func = np.ones(num+1)
    time_spline = np.zeros(num+1)
    delta = (t1-t0)/num
    i = 0
    n = t0
    #print(n)
    #print(t1)
    #print(f't0 = {t0}, loop ends when n >= {np.round(t1, 4)}')
    while n < t1 + delta/100:
        func[i] = hermite_basis(n)[0, :].dot(coeff)
        time_spline[i] = n
        #print(f'array func at index {i} has the value of {func[i]}, n = {n}')
        i = i + 1
        #print(i)
        n = n + delta
        #print(n)
    #print(f'loop breaks at i = {i}, n = {n}')
    return func, time_spline

def quintic_hermite_approx(num, joint_n, step_n):

    rdr = generate_traj(num)
    func = np.zeros((num, step_n+1))
    time_spline = np.zeros((num, step_n+1))

    for i in range(num):
        x_point = np.array([rdr[joint_n].q[i], rdr[joint_n].qd[i], rdr[joint_n].qdd[i]])
        y_point = np.array([rdr[joint_n].q[i+1], rdr[joint_n].qd[i+1], rdr[joint_n].qdd[i+1]])
        func[i, :] = solver(rdr[6][i], rdr[6][i+1], x_point, y_point, step_n)[0]
        time_spline[i, :] = solver(rdr[6][i], rdr[6][i+1], x_point, y_point, step_n)[1]

    flat_func = func.flatten()
    flat_time_spline = time_spline.flatten()
    print(func.flatten())
    print(time_spline.flatten())

    plt.plot(flat_time_spline, flat_func)
    plt.plot(rdr[6], np.round(rdr[joint_n].q, 4))
    plt.show()

quintic_hermite_approx(20, 3, 40)

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