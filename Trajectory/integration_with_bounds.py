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
from scipy.interpolate import interp1d
from tolerance import *

traj_time = 2
joint = generate_traj_time(traj_time) # generate trapezoidal trajectory with given trajectory time, discretized with 100 time steps, change config in traj.py
time = joint[6]
tolerance_band = np.zeros((2, 100*traj_time+3, 6))

for i in range(6):
    angle = joint[i].q
    velocity = joint[i].qd
    upper = create_tolerance_bands(angle, velocity, time, 0.5, "upper")
    lower = create_tolerance_bands(angle, velocity, time, 0.5, "lower")
    tolerance_band[0, :, i] = upper[1]
    tolerance_band[1, :, i] = lower[1]

print(time)

# Below is a whole section of image output....
"""
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(time, joint[0].q, label='original joint1', color='red')
ax1.plot(upper[0], tolerance_band[0, :, 0], label='tolerance band', linestyle='dashed', color='green')
ax1.plot(upper[0], tolerance_band[1, :, 0], linestyle='dashed', color='green')
ax1.legend()
ax2.plot(time, joint[1].q, label='original joint2', color='red')
ax2.plot(upper[0], tolerance_band[0, :, 1], label='tolerance band', linestyle='dashed', color='green')
ax2.plot(upper[0], tolerance_band[1, :, 1], linestyle='dashed', color='green')
ax2.legend()
ax3.plot(time, joint[2].q, label='original joint3', color='red')
ax3.plot(upper[0], tolerance_band[0, :, 2], label='tolerance band', linestyle='dashed', color='green')
ax3.plot(upper[0], tolerance_band[1, :, 2], linestyle='dashed', color='green')
ax3.legend()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(time, joint[3].q, label='original joint1', color='red')
ax1.plot(upper[0], tolerance_band[0, :, 3], label='tolerance band', linestyle='dashed', color='green')
ax1.plot(upper[0], tolerance_band[1, :, 3], linestyle='dashed', color='green')
ax1.legend()
ax2.plot(time, joint[4].q, label='original joint2', color='red')
ax2.plot(upper[0], tolerance_band[0, :, 4], label='tolerance band', linestyle='dashed', color='green')
ax2.plot(upper[0], tolerance_band[1, :, 4], linestyle='dashed', color='green')
ax2.legend()
ax3.plot(time, joint[5].q, label='original joint3', color='red')
ax3.plot(upper[0], tolerance_band[0, :, 5], label='tolerance band', linestyle='dashed', color='green')
ax3.plot(upper[0], tolerance_band[1, :, 5], linestyle='dashed', color='green')
ax3.legend()

plt.show()
"""