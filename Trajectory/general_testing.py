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
from scipy.stats import truncnorm
from scipy.integrate import simpson
from plot_traj import *
from traj import *
from HKA_kalman_gain import *
from energy_est import *
import time as ti
from tau import *
"""
start_time1 = ti.time()
a = cal_tau(np.ones(6), np.ones(6), np.ones(6))
end_time1 = ti.time()
t1 = end_time1 - start_time1

start_time2 = ti.time()
b = calc_tau(np.ones(6), np.ones(6), np.ones(6))
end_time2 = ti.time()
t2 = end_time2 - start_time2

print(t1)
print(t2)
"""
Yu = rtb.models.DH.Yu()
traj = generate_traj_time(2.5, 201)
angle = np.zeros((6, 201))
velocity = np.zeros((6, 201))
accel = np.zeros((6, 201))
mass_vec = np.zeros(6)

for j in range(6):
    angle[j, :] = traj[j].q
    velocity[j, :] = traj[j].qd
    accel[j, :] = traj[j].qdd

ee_list = np.zeros((3, 201))
for i in range(201):
    ee = np.array(Yu.fkine(angle[:, i]))
    ee = ee[:, -1]
    ee = ee[:3]
    ee_list[:, i] = ee

xline2 = ee_list[0, :]
yline2 = ee_list[1, :]
zline2 = ee_list[2, :]


ax = plt.axes(111, projection='3d')
#ax.set_xlim([-8, 8])
#ax.set_ylim([-8, 8])
#ax.set_zlim([-1, 1])
ax.set_xlabel('x coordinate in m')
ax.set_ylabel('Y coordinate in m')
ax.set_zlabel('Z coordinate in m')
#ax.plot3D(xline, yline, zline)
#ax.plot3D(xline1, yline1, zline1, color='blue')
ax.plot3D(xline2, yline2, zline2, color='red', linewidth=1, label='Trajectory of center of mass')
plt.show()

print(ee_list)
