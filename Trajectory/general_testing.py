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

