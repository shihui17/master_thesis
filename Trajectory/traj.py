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
"""
def generate_traj(num):
    Yu = rtb.models.DH.Yu()
    q_end = Yu.qa
    t = np.linspace(0, num, num)
    tg1 = tools.trapezoidal(pi/2, q_end[0], num)
    tg2 = tools.trapezoidal(pi/3, q_end[1], num)
    tg3 = tools.trapezoidal(pi/6, pi, t)
    tg4 = tools.trapezoidal(pi/6, q_end[3], num)
    tg5 = tools.trapezoidal(pi/3, q_end[4], num)
    tg6 = tools.trapezoidal(pi/6, q_end[5], num)
    return tg1, tg2, tg3, tg4, tg5, tg6, t
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
"""

def generate_traj_time(traj_time):
    Yu = rtb.models.DH.Yu()
    t = np.linspace(0, traj_time, 201)
    tg1 = tools.trapezoidal(-pi/2, pi/6, t)
    tg2 = tools.trapezoidal(-pi/2, pi/2, t)
    tg3 = tools.trapezoidal(pi/3, -pi, t)
    tg4 = tools.trapezoidal(-pi/3, pi/2, t)
    tg5 = tools.trapezoidal(-pi/3, pi/3, t)
    tg6 = tools.trapezoidal(0, -pi, t)

    return tg1, tg2, tg3, tg4, tg5, tg6, t

"""

p1 = 5
p2 = 10
p3 = 15
p4 = 20

def generate_vectors(tuple, pt):
    joint_q = np.zeros(6)
    joint_qd = np.zeros(6)
    joint_qdd = np.zeros(6)
    for i in range(6):
        joint_q[i] = tuple[i].q[pt]
        joint_qd[i] = tuple[i].qd[pt]
        joint_qdd[i] = tuple[i].qdd[pt]
    return joint_q, joint_qd, joint_qdd

# evaluate the power output of each robot axis at a given point eval_pt on the trajectory
def eval_power(joint_vec):
    # calculate the torque vector(tau), argument in the order of: q1, qd1, qdd1, q2, ..... , qdd6
    torque_vec = np.abs(cal_tau(joint_vec[0], joint_vec[1], joint_vec[2]))
    #print(np.round(torque_vec, 2))
    # calculate the velocity vector(qd)
    velocity_vec = joint_vec[1]
    #print(velocity_vec)
    # calculate power output, elementwise multiplication of torque vector and velocity vector
    power = np.multiply(torque_vec, velocity_vec)
    #print(power)
    return power


"""

# The following section is for graph generation. Uncomment to visualise q, dq and ddq
"""
rdr = generate_traj_time(2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(rdr[6], np.round(rdr[0].q, 6), label='q1')
ax1.plot(rdr[6], np.round(rdr[1].q, 6), label='q2')
ax1.plot(rdr[6], np.round(rdr[2].q, 6), label='q3')
ax1.plot(rdr[6], np.round(rdr[3].q, 6), label='q4')
ax1.plot(rdr[6], np.round(rdr[4].q, 6), label='q5')
ax1.plot(rdr[6], np.round(rdr[5].q, 6), label='q6')
ax1.set_xlabel('Travel Time in s')
ax1.set_ylabel('joint angle in rad')
ax1.legend()

ax2.plot(rdr[6], np.round(rdr[0].qd, 6), label='q1')
ax2.plot(rdr[6], np.round(rdr[1].qd, 6), label='q2')
ax2.plot(rdr[6], np.round(rdr[2].qd, 6), label='q3')
ax2.plot(rdr[6], np.round(rdr[3].qd, 6), label='q4')
ax2.plot(rdr[6], np.round(rdr[4].qd, 6), label='q5')
ax2.plot(rdr[6], np.round(rdr[5].qd, 6), label='q6')
ax2.set_xlabel('Travel Time in s')
ax2.set_ylabel('joint velocity in 1/s')
ax2.legend()

ax3.plot(rdr[6], np.round(rdr[0].qdd, 6), label='q1')
ax3.plot(rdr[6], np.round(rdr[1].qdd, 6), label='q2')
ax3.plot(rdr[6], np.round(rdr[2].qdd, 6), label='q3')
#ax3.plot(rdr[6][0], rdr[2].qdd[0], 'o')
ax3.plot(rdr[6], np.round(rdr[3].qdd, 6), label='q4')
ax3.plot(rdr[6], np.round(rdr[4].qdd, 6), label='q5')
ax3.plot(rdr[6], np.round(rdr[5].qdd, 6), label='q6')
ax3.set_xlabel('Travel Time in s')
ax3.set_ylabel('joint acceleration in $1/s^2$')
ax3.set_ylim(bottom=-5, top=5)
ax3.legend()

plt.show()
"""
#fig1 = plt.figure()
#fig1 = plt.plot(rdr[6], np.round(rdr[0].q, 6))
#plt.xlabel('Trajectory Time in s')
#plt.ylabel('Joint angle in rad')
#plt.show()
