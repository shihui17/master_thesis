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

def generate_traj_time(traj_time):
    Yu = rtb.models.DH.Yu()
    q_end = Yu.qa
    t = np.linspace(0, traj_time, 101)
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

for sn in range(sample_num):

    energy = 0
    mu_index = math.floor(100/(sample_num+1)*(sn+1))

    while ctr < mu_index:


        for jn in range(6):

            array_q[jn] = rdr[jn].q[ctr]
            array_qd[jn] = rdr[jn].qd[ctr]
            array_qdd[jn] = rdr[jn].qdd[ctr]

        torq = cal_tau(array_q, array_qd, array_qdd)
        power = np.multiply(torq, array_qd)
        energy = energy + np.linalg.norm(power, 1)*dt
        ctr = ctr +1
    print(energy)
        