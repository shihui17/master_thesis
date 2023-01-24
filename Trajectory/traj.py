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

Yu = rtb.models.DH.Yu()
q0 = Yu.qa
efe = Yu.fkine(q0)
sol = Yu.ikine_LM(efe)

q_begin = np.zeros(6)
q_end = sol.q

t = np.linspace(0, 6, num=30)

tg1 = tools.trapezoidal(0, q_end[0], 30)
tg2 = tools.trapezoidal(0, q_end[1], 30)
tg3 = tools.trapezoidal(0, q_end[2], 30)
tg4 = tools.trapezoidal(0, q_end[3], 30)
tg5 = tools.trapezoidal(0, q_end[4], 30)
tg6 = tools.trapezoidal(0, q_end[5], 30)

p1 = 5
p2 = 10
p3 = 15
p4 = 20

# evaluate the power output of each robot axis at a given point eval_pt on the trajectory
def eval_power(eval_pt):
    # calculate the torque vector(tau), argument in the order of: q1, qd1, qdd1, q2, ..... , qdd6
    torque_vec = np.abs(cal_tau(tg1.q[eval_pt-1], tg1.qd[eval_pt-1], tg1.qdd[eval_pt-1], tg2.q[eval_pt-1], tg2.qd[eval_pt-1], tg2.qdd[eval_pt-1], 
    tg3.q[eval_pt-1], tg3.qd[eval_pt-1], tg3.qdd[eval_pt-1], tg4.q[eval_pt-1], tg4.qd[eval_pt-1], tg4.qdd[eval_pt-1], 
    tg5.q[eval_pt-1], tg5.qd[eval_pt-1], tg5.qdd[eval_pt-1], tg6.q[eval_pt-1], tg6.qd[eval_pt-1], tg6.qdd[eval_pt-1]))
    #print(np.round(torque_vec, 2))
    # calculate the velocity vector(qd)
    velocity_vec = np.array([tg1.qd[eval_pt-1], tg2.q[eval_pt-1], tg3.qd[eval_pt-1], tg4.qd[eval_pt-1], tg5.qd[eval_pt-1], tg6.qd[eval_pt-1]])
    #print(velocity_vec)
    # calculate power output, elementwise multiplication of torque vector and velocity vector
    power = np.multiply(torque_vec, velocity_vec)
    print(power)
    return power

eval_power(p1)

# The following section is for graph generation. Uncomment to visualise q, dq and ddq
""" 
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained',)

ax1.plot(t, np.round(tg1.q, 6), label='q1')
ax1.plot(t, np.round(tg2.q, 6), label='q2')
ax1.plot(t, np.round(tg3.q, 6), label='q3')
ax1.plot(t, np.round(tg4.q, 6), label='q4')
ax1.plot(t, np.round(tg5.q, 6), label='q5')
ax1.plot(t, np.round(tg6.q, 6), label='q6')
ax1.set_xlabel('Travel Time in s')
ax1.set_ylabel('joint angle in rad')
ax1.legend()

ax2.plot(t, np.round(tg1.qd, 6), label='q1')
ax2.plot(t, np.round(tg2.qd, 6), label='q2')
ax2.plot(t, np.round(tg3.qd, 6), label='q3')
ax2.plot(t, np.round(tg4.qd, 6), label='q4')
ax2.plot(t, np.round(tg5.qd, 6), label='q5')
ax2.plot(t, np.round(tg6.qd, 6), label='q6')
ax2.set_xlabel('Travel Time in s')
ax2.set_ylabel('joint velocity in 1/s')
ax2.legend()

ax3.plot(t, np.round(tg1.qdd, 6), label='q1')
ax3.plot(t, np.round(tg2.qdd, 6), label='q2')
ax3.plot(t, np.round(tg3.qdd, 6), label='q3')
ax3.plot(t, np.round(tg4.qdd, 6), label='q4')
ax3.plot(t, np.round(tg5.qdd, 6), label='q5')
ax3.plot(t, np.round(tg6.qdd, 6), label='q6')
ax3.set_xlabel('Travel Time in s')
ax3.set_ylabel('joint acceleration in $1/s^2$')
ax3.legend()

plt.show()
"""