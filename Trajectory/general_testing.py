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

q = np.loadtxt("result_q_int.txt")
qd = np.loadtxt("result_qd_int.txt")
qdd = np.loadtxt("result_qdd_int.txt")
time = np.loadtxt("time_vec_int.txt")

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
def poly4_interp(x):
    A_matrix = np.array([[x**4, x**3, x**2, x, 1], [4 * x**3, 3 * x**2, 2 * x, 1, 0], [12 * x**2, 6 * x, 2, 0, 0]])
    return A_matrix

A_matrix1 = poly4_interp(time[-1])
print(A_matrix1)
A_matrix2 = poly4_interp(2)
A_matrix2 = np.delete(A_matrix2, 2, 0)
print(A_matrix2)
A_matrix_whole = np.vstack((A_matrix1, A_matrix2))
print(A_matrix_whole)

B_vec = np.array([q[-1, 0], qd[-1, 0], qdd[-1, 0], 1.570796326794895670e+00, 0])
print(B_vec)

sol = np.linalg.solve(A_matrix_whole, B_vec)
print(sol)

def poly4(a, b, c, d, e, x):
    func = a * x**4 + b * x**3 + c * x**2 + d * x + e
    return func

tt = np.linspace(time[-1], 2, num=100)
yt = [poly4(sol[0], sol[1], sol[2], sol[3], sol[4], t) for t in tt]

def poly4_d(a, b, c, d, x):
    func = 4 * a * x**3 + 3* b * x**2 + 2 * c * x + d
    return func

yt_d = [poly4_d(sol[0], sol[1], sol[2], sol[3], t) for t in tt]

def poly4_dd(a, b, c, x):
    func = 12 * a * x**2 + 6* b * x + 2 * c
    return func

yt_dd = [poly4_dd(sol[0], sol[1], sol[2], t) for t in tt]

t_j1 = np.concatenate((t_list, tt))
q_j1 = np.concatenate((q_total[0], yt))

qd_j1 = np.concatenate((qd_total[0], yt_d))
qdd_j1 = np.concatenate((qdd_total[0], yt_dd))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')

fig.suptitle(f'Trajecotry of joint 1', fontsize=16)
ax1.plot(joint[6], joint[0].q, color='blue', label='Original angle')
ax1.plot(time, q[:, 0], 'r+', label='Optimized angle')
ax1.plot(t_j1, q_j1, color='green', label='Optimized angle profile')
ax1.legend()

ax2.plot(joint[6], joint[0].qd, color='blue', label='Original velocity')
ax2.plot(time, qd[:, 0], 'r+', label='Optimized velocity')
ax2.plot(t_j1, qd_j1, color='green', label='Optimized velocity profile')
ax2.legend()
ax3.plot(joint[6], joint[0].qdd, color='blue', label='Original acceleration')
ax3.plot(time, qdd[:, 0], 'r+', label='Optimized acceleration')
ax3.plot(t_j1, qdd_j1, color='green', label='Optimized accel. profile')
ax3.legend()

plt.show()