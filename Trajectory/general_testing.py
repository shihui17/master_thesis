import roboticstoolbox as rtb
import numpy as np
import math
from math import pi
from roboticstoolbox import tools as tools
import spatialmath as sm
import spatialgeometry as sg
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import matlib
from call_tau import *
from traj import *
from scipy.stats import truncnorm
from scipy.interpolate import interp1d


"""
a = generate_traj(20)
power = np.zeros((20, 6))
sum_power = np.zeros(20)

for i in range(20):
    joint_vec = generate_vectors(a, i)
    power[i, :] = eval_power(joint_vec)
    sum_power[i] = np.linalg.norm(power[i, :], 1)

power01 = power[:, 0]
power02 = power[:, 1]
power03 = power[:, 2]
power04 = power[:, 3]
power05 = power[:, 4]
power06 = power[:, 5]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained',)
ax1.plot(a[6], np.round(power01, 6), label='q1')
ax1.plot(a[6], np.round(power02, 6), label='q2')
ax1.plot(a[6], np.round(power03, 6), label='q3')
ax1.plot(a[6], np.round(power04, 6), label='q4')
ax1.plot(a[6], np.round(power05, 6), label='q5')
ax1.plot(a[6], np.round(power06, 6), label='q6')
ax1.plot(a[6], sum_power, label='Q')
ax1.set_xlabel('Travel Time in s')
ax1.set_ylabel('joint angle in rad')
ax1.legend()

plt.show()

"""
"""
def generate_traj_time(traj_time):
    Yu = rtb.models.DH.Yu()
    q_end = Yu.qa
    t = np.linspace(0, traj_time, 201)
    tg1 = tools.trapezoidal(pi/2, q_end[0], t)
    tg2 = tools.trapezoidal(pi/3, q_end[1], t)
    tg3 = tools.trapezoidal(pi/6, pi, t)
    tg4 = tools.trapezoidal(pi/6, q_end[3], t)
    tg5 = tools.trapezoidal(pi/3, q_end[4], t)
    tg6 = tools.trapezoidal(pi/6, q_end[5], t)

    return tg1, tg2, tg3, tg4, tg5, tg6, t

rdr = generate_traj_time(2)
array_q = np.zeros(6)
array_qd = np.zeros(6)
array_qdd = np.zeros(6)
dt = rdr[6][1] - rdr[6][0]
energy = 0
sample_num = 4
ctr = 0
#print(rdr[6])
#print(len(rdr[6]))
index = np.where(np.round(rdr[6], 2) == 0.35)
print(rdr[0].q)
"""
""""""

from lerp import *

qdd = np.array([[-1.76714587e+00, -2.94524311e+00,  2.94524311e+00, -2.35619449e+00, -2.94524311e+00, -5.89048623e-01],
                [-8.83573391e-01, -1.47262158e+00,  2.06556155e+00, -1.17809990e+00, -1.47262188e+00, -5.25396973e-01],
                [-1.37743004e+00, -1.47262156e+00,  4.02678813e+00, -3.53429156e+00, -4.41786467e+00, -6.85241066e-01],
                [-1.06081292e-06, -1.42771274e-07,  1.60353320e-06, -4.52318512e-06, -9.48333337e-08, -1.08932988e-05],
                [-6.33714154e-09, -4.88211313e-09, -8.73540898e-08, -9.91883478e-08, -5.37750218e-08, -6.49061527e-08],
                [ 8.83573215e-01,  1.47262181e+00, -4.41786387e+00,  1.79105292e+00,  1.47263890e+00,  6.05149017e-01],
                [ 8.83573292e-01,  1.47262185e+00, -1.47262285e+00,  2.48478931e+00,  4.41785654e+00,  5.11819288e-01]])

qd = np.array([[-0.,         -0.,          0.,         -0.,         -0.,         -0.,       ],
               [-0.37867418, -0.63112353,  0.71582924, -0.5048992,  -0.63112357, -0.15920651],
               [-0.70167467, -1.05187255,  1.5861649,  -1.17809798, -1.47262165, -0.3321548 ],
               [-0.89845054, -1.26224708,  2.16142058, -1.68299742, -2.10374519, -0.43004794],
               [-0.89845069, -1.2622471,   2.1614208,  -1.68299808, -2.10374521, -0.43004951],
               [-0.77222595, -1.05187255,  1.53029737, -1.42713339, -1.89336823, -0.34359966],
               [-0.51977645, -0.63112346,  0.68879927, -0.81629878, -1.05186888, -0.18403276]])

q = np.array([[ 1.57079633,  1.04719755,  0.52359878,  0.52359878,  1.04719755,  0.52359878],
 [ 1.51068932,  0.94701921,  0.63184432,  0.44345607,  0.94701921,  0.50042198],
 [ 1.35971334,  0.7065912 ,  0.94735895 , 0.21905642,  0.66651984,  0.43131488],
 [ 1.12175376,  0.36598485,  1.51012141, -0.21371427,  0.12555679,  0.31776735],
 [ 0.86505357,  0.00534283,  2.12767019, -0.6945708,  -0.47551326,  0.19489621],
 [ 0.62037478, -0.33526353,  2.685112,   -1.15105932, -1.05654742,  0.08025824],
 [ 0.43580301, -0.57569153,  2.98209015, -1.47626893, -1.49733107,  0.00551708]])

time = np.array([0.,         0.28571429, 0.57142857, 0.85714286, 1.14285714, 1.42857143,
 1.71428571])

qdd_plot = []
t_list = []
qd_plot = []
q_plot = []
q_total = []
qd_total = []
qdd_total = []

for j in range(6):

    for ctr, t in enumerate(time):
        qdd1 = qdd[ctr, j]
        qdd2 = qdd[ctr+1, j]
        t1 = time[ctr]
        t2 = time[ctr+1]
        t_eval = np.linspace(t1, t2, num = 10)
        t_list.append(t_eval)
        qdd_plot.append([lerp_func(t1, t2, qdd1, qdd2, t) for t in t_eval])
        if ctr == len(time) - 2: break
    
#print(qdd_plot)
qdd_j1 = np.array(qdd_plot[0:6])
qdd_j1 = qdd_j1.flatten()
qdd_total.append(qdd_j1)
qdd_j2 = np.array(qdd_plot[6:12])
qdd_j2 = qdd_j2.flatten()
qdd_total.append(qdd_j2)
qdd_j3 = np.array(qdd_plot[12:18])
qdd_j3 = qdd_j3.flatten()
qdd_total.append(qdd_j3)
qdd_j4 = np.array(qdd_plot[18:24])
qdd_j4 = qdd_j4.flatten()
qdd_total.append(qdd_j4)
qdd_j5 = np.array(qdd_plot[24:30])
qdd_j5 = qdd_j5.flatten()
qdd_total.append(qdd_j5)
qdd_j6 = np.array(qdd_plot[30:36])
qdd_j6 = qdd_j6.flatten()
qdd_total.append(qdd_j6)

t_list = np.array(t_list[0:6])
#print(qdd_j6)
t_list = t_list.flatten()
#print(t_list)
#qdd_total = np.concatenate(qdd_j1, qdd_j2, qdd_j3, qdd_j4, qdd_j5, qdd_j6)

for j in range(6):

    for ctr, t in enumerate(time):
        qdd1 = qdd[ctr, j]
        qdd2 = qdd[ctr+1, j]
        t1 = time[ctr]
        t2 = time[ctr+1]
        t_eval = np.linspace(t1, t2, num = 10)
        qd_plot.append([lerp_func_integral(t1, t2, qdd1, qdd2, t, qd0=qd[ctr, j]) for t in t_eval])
        if ctr == len(time) - 2: break

qd_j1 = np.array(qd_plot[0:6])
qd_j1 = qd_j1.flatten()
qd_total.append(qd_j1)
qd_j2 = np.array(qd_plot[6:12])
qd_j2 = qd_j2.flatten()
qd_total.append(qd_j2)
qd_j3 = np.array(qd_plot[12:18])
qd_j3 = qd_j3.flatten()
qd_total.append(qd_j3)
qd_j4 = np.array(qd_plot[18:24])
qd_j4 = qd_j4.flatten()
qd_total.append(qd_j4)
qd_j5 = np.array(qd_plot[24:30])
qd_j5 = qd_j5.flatten()
qd_total.append(qd_j5)
qd_j6 = np.array(qd_plot[30:36])
qd_j6 = qd_j6.flatten()
qd_total.append(qd_j6)


for j in range(6):

    for ctr, t in enumerate(time):
        qdd1 = qdd[ctr, j]
        qdd2 = qdd[ctr+1, j]
        t1 = time[ctr]
        t2 = time[ctr+1]
        t_eval = np.linspace(t1, t2, num = 10)
        q_plot.append([lerp_func_double_integral(t1, t2, qdd1, qdd2, t, qd0=qd[ctr, j], q0=q[ctr, j]) for t in t_eval])
        if ctr == len(time) - 2: break

#print(q_plot)
q_j1 = np.array(q_plot[0:6])
q_j1 = q_j1.flatten()
q_total.append(q_j1)
q_j2 = np.array(q_plot[6:12])
q_j2 = q_j2.flatten()
q_total.append(q_j2)
q_j3 = np.array(q_plot[12:18])
q_j3 = q_j3.flatten()
q_total.append(q_j3)
q_j4 = np.array(q_plot[18:24])
q_j4 = q_j4.flatten()
q_total.append(q_j4)
q_j5 = np.array(q_plot[24:30])
q_j5 = q_j5.flatten()
q_total.append(q_j5)
q_j6 = np.array(q_plot[30:36])
q_j6 = q_j6.flatten()
q_total.append(q_j6)

#q_total = np.concatenate(q_j1, q_j2, q_j3, q_j4, q_j5, q_j6)
joint = generate_traj_time(2)

for j_num in range(6):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
    fig.suptitle(f'Trajecotry of joint {j_num+1}', fontsize=16)
    ax1.plot(joint[6], joint[j_num].q, color='blue', label='Original angle')
    ax1.plot(time, q[:, j_num], 'r+', label='Optimized angle')
    ax1.plot(t_list, q_total[j_num], color='green', label='Optimized angle profile')
    ax1.legend()
    ax2.plot(joint[6], joint[j_num].qd, color='blue', label='Original velocity')
    ax2.plot(time, qd[:, j_num], 'r+', label='Optimized velocity')
    ax2.plot(t_list, qd_total[j_num], color='green', label='Optimized velocity profile')
    ax2.legend()
    ax3.plot(joint[6], joint[j_num].qdd, color='blue', label='Original acceleration')
    ax3.plot(time, qdd[:, j_num], 'r+', label='Optimized acceleration')
    ax3.plot(t_list, qdd_total[j_num], color='green', label='Optimized accel. profile')
    ax3.legend()

    for a, b in zip(time, np.round(qdd[:, j_num], 3)):
        plt.text(a, b, str(b))

plt.show()