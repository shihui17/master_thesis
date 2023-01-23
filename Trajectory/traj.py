import roboticstoolbox as rtb
import numpy as np
import math
from math import pi
from roboticstoolbox import tools as tools
import spatialmath as sm
import spatialgeometry as sg
import matplotlib as mpl
import matplotlib.pyplot as plt

Yu = rtb.models.DH.Yu()
q0 = Yu.qa
print(q0)
efe = Yu.fkine(q0)
sol = Yu.ikine_LM(efe)

q_begin = np.zeros(6)
q_end = sol.q

t = np.arange(1, 31)

tg1 = tools.trapezoidal(0, q_end[0], 30)
tg2 = tools.trapezoidal(0, q_end[1], 30)
tg3 = tools.trapezoidal(0, q_end[2], 30)
tg4 = tools.trapezoidal(0, q_end[3], 30)
tg5 = tools.trapezoidal(0, q_end[4], 30)
tg6 = tools.trapezoidal(0, q_end[5], 30)

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

#plt.show()

n = len(tg1.q)
print(n)