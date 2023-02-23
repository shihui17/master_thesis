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
from HKA_Alg import heuristic_kalman
from scipy.interpolate import BPoly


def quintic_hermite_interpolation(x, x1, x2, p0, v0, a0, p1, v1, a1):
    t = (x - x1)/(x2 - x1)
    
    h0 = -6*t**5 + 15*t**4 - 10*t**3 + 1
    h1 = (-3*t**5 + 8*t**4 - 6*t**3 + t) * (x2 - x1)
    h2 = (-0.5*t**5 + 1.5*t**4 -1.5*t**3 +0.5*t**2) * (x2 - x1)
    h3 = 6*t**5 - 15*t**4 + 10*t**3
    h4 = (-3*t**5 + 7*t**4 -4*t**3) * (x2 - x1)
    h5 = (0.5*t**5 - t**4 +0.5*t**3) * (x2 - x1)

    h10 = (-30*t**4 + 60*t**3 - 30*t**2) / (x2 - x1)
    h11 = (-15*t**4 + 32*t**3 - 18*t**2 + 1) #* (x2 - x1)
    h12 = (-2.5*t**4 + 6*t**3 - 4.5*t**2 + t) * (x2 - x1)
    h13 = (30*t**4 - 60*t**3 + 30*t**2)  / (x2 - x1)
    h14 = (-15*t**4 + 28*t**3 - 12*t**2) #* (x2 - x1)
    h15 = (2.5*t**4 - 4*t**3 + 1.5*t**2) * (x2 - x1)

    h20 = (-120*t**3 + 180*t**2 -60*t) / (x2 - x1)**2
    h21 = (-60*t**3 + 96*t**2 -36*t) / (x2 - x1)
    h22 = (-10*t**3 + 18*t**2 - 9*t + 1) #* (x2 - x1)
    h23 = (120*t**3 - 180*t**2 + 60*t) / (x2 - x1)**2
    h24 = (-60*t**3 + 84*t**2 - 24*t) / (x2 - x1)
    h25 = (10*t**3 - 12*t**2 + 3*t) #* (x2 - x1)


    joint_q = h0 * p0 + h1 * v0 + h2 * a0 + h3 * p1 + h4 * v1 + h5 * a1   
    joint_qd = h10 * p0 + h11 * v0 + h12 * a0 + h13 * p1 + h14 * v1 + h15 * a1   
    joint_qdd = h20 * p0 + h21 * v0 + h22 * a0 + h23 * p1 + h24 * v1 + h25 * a1


    #print(f'for joint_qd: {h10}, {p0}, {h11}, {v0}, {h12}, {a0}, {h13}, {p1}, {h14}, {v1}, {h15}, {a1}\n\n')
    #print(t)

    return joint_q, joint_qd, joint_qdd

#result_assemble = heuristic_kalman(50, 5, np.array([[0, 0, 0, 0, 0, 0], [0.0003, 5, 6, 8, 20, 10]]), 0, 0, 6, 40, 20)

#result_q = result_assemble[0]
#result_qd = result_assemble[1]
#result_qdd = result_assemble[2]

#q_path = np.array([0, 1, 2, 3, 4, -5])
#qd_path = np.array([0.2, 0.4, 0, 0, 0, -1])
#qdd_path = np.array([8.69452293e-03, 9.58848730e-03, 2.80075542e-03, 4.32528318e-05, -8.31117476e-03, -8.74073843e-03])

##q_path = np.array([0.80334404, 0.90994253, 1.08698838, 1.2762963, 1.45183501, 1.55265356])
##qd_path = np.array([0.00928984, 0.04634367, 0.05832201, 0.05281152, 0.03815433, 0.00849902])
#qdd_path = np.array([8.69452293e-03, 9.58848730e-03, 2.80075542e-03, 4.32528318e-05, -8.31117476e-03, -8.74073843e-03])

#time = np.array([2, 5, 8, 11, 14, 17])


def evaluate(joint_num):
    
    result_assemble = heuristic_kalman(150, 15, np.array([[0, 0, 0, 0, 0, 0], [0.0003, 5, 6, 8, 20, 10]]), 0, 0, 6, 6, 10)
    result_q = result_assemble[0]
    result_qd = result_assemble[1]
    result_qdd = result_assemble[2]
    time = result_assemble[3]
    time = time.flatten()
    #print(f'TEST! time array = {time}')
    og_data = result_assemble[4]

    q_path = result_q[:, joint_num] 
    qd_path = result_qd[:, joint_num]
    qdd_path = result_qdd[:, joint_num]



    q_assemble = np.vstack((q_path, qd_path, qdd_path))
    q_assemble = np.transpose(q_assemble)
    print(q_assemble)
    func = BPoly.from_derivatives(time, q_assemble)
    t_eval = np.linspace(time[0], time[-1], 100)
    y = func(t_eval)
    y_d = func.derivative()(t_eval)
    y_dd = func.derivative(2)(t_eval)

    curve = list()
    curve_d = list()
    curve_dd = list()

    for n in range(len(q_path)-1):
        t_values = np.linspace(time[n], time[n+1], num=50)
        #print(f'TEST! n = {n}')   
        if n == 0:
            t_total = t_values
        else:
            t_total = np.concatenate((t_total, t_values))
        curve.append( [quintic_hermite_interpolation(x, time[n], time[n+1], q_path[n], qd_path[n], qdd_path[n], q_path[n+1], qd_path[n+1], qdd_path[n+1])[0] for x in t_values] )
        curve_d.append( [quintic_hermite_interpolation(x, time[n], time[n+1], q_path[n], qd_path[n], qdd_path[n], q_path[n+1], qd_path[n+1], qdd_path[n+1])[1] for x in t_values] )
        curve_dd.append( [quintic_hermite_interpolation(x, time[n], time[n+1], q_path[n], qd_path[n], qdd_path[n], q_path[n+1], qd_path[n+1], qdd_path[n+1])[2] for x in t_values] )

    curve_arr = np.array(curve)
    curve_d_arr = np.array(curve_d)
    curve_dd_arr = np.array(curve_dd)
    curve_arr = curve_arr.flatten()
    curve_d_arr = curve_d_arr.flatten()
    curve_dd_arr = curve_dd_arr.flatten()

    #print(curve_d)
    #print(time)
    #print(t_total)

    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, layout = 'constrained')
    ax1.plot(t_total, curve_arr, label = 'Quintic Hermite')
    ax1.plot(t_eval, y, label = 'Berstein')
    ax1.plot(og_data[6], og_data[joint_num].q)
    ax2.plot(t_total, curve_d_arr)
    ax2.plot(og_data[6], og_data[joint_num].qd)
    ax2.legend()
    ax2.plot(t_eval, y_d, label = 'First Derivative')
    ax3.plot(t_total, curve_dd_arr)
    ax3.plot(og_data[6], og_data[joint_num].qdd)
    ax3.plot(t_eval, y_dd, label = 'Second Derivative')
    ax3.legend()
    plt.show()
    
    #print(og_data[2].qdd)
    
eval = evaluate(2)
#print(f'time series: {eval[1]}\n\npoints: {eval[0]}')""
