'''
@author: Shihui Liu

Simple implementation of Particle Swarm optimization using pyswarms package. The same objective function as in 08_Comparison.py is used. Optimization problem formulated based on velocity profile 
optimization method.
'''

from math import pi
import numpy as np
from roboticstoolbox import tools as tools
import matplotlib.pyplot as plt
from call_tau import *
from traj import *
from HKA_kalman_gain import *
from energy_est import *
import time as ti
from scipy.optimize import minimize
from plot_traj import *
import time as ti
import pyswarms as ps
from utility import *

def energy_consumption(qdd):

    n_particles = qdd.shape[1]
    q0 = np.zeros(n_particles)
    qf = np.zeros(n_particles)
    step = 201
    angle = np.zeros((n_particles, step)) 
    velocity = np.zeros((n_particles, step))
    accel = np.zeros((n_particles, step))
    vel_max = np.zeros(n_particles)
    t_accel = np.zeros(n_particles) 
    t_brake = np.zeros(n_particles)
    angle_array = np.zeros((n_particles, 201))
    velocity_array = np.zeros((n_particles, 201))
    accel_array = np.zeros((n_particles, 201))
    acceleration_time = np.zeros(n_particles)
    flag = np.zeros(n_particles)
    t_min = np.zeros(n_particles) 
    t_max = np.zeros(n_particles) 
    a = np.zeros(n_particles)
    start1 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
    end1 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])
    trajectory_info = generate_traj_time(2, step, start1, end1)
    time = trajectory_info[6]
    tf = time[-1] # finish
    for j in range(n_particles): # reads necessary data from the original trajectory
        angle[j, :] = trajectory_info[j].q
        velocity[j, :] = trajectory_info[j].qd
        accel[j, :] = trajectory_info[j].qdd
        q0[j] = angle[j, 0]
        qf[j] = angle[j, -1]
        if q0[j] == qf[j]: # if the joint does not move originally, then prevent the optimizer from moving it during optimization
            vel_max[j] = 0
            t_accel[j] = 0
            t_brake[j] = tf
            a[j] = 0
            flag[j] = 1
        else: 
            vel_max[j] = (qf[j] - q0[j]) / tf * 1.5 
            t_accel[j] = np.round((q0[j] - qf[j] + vel_max[j] * tf) / vel_max[j], 2) # end of acceleration, rounded to 2 decimals to exactly match the time points in traj[6]
            t_brake[j] = np.round(tf - t_accel[j], 2) # start of braking
            a[j] = vel_max[j] / t_accel[j]
            t_min[j] = abs(vel_max[j] / 100) # boundary condition: maximal angular acceleration 100 s^-2
            t_max[j] = abs(tf - tf / 3 - vel_max[j] / 100) # maximal allowed acceleration time is reached when the joint has to brake with the maximal angular acceleration in order to reach end configuration (100 s^-2)
   
    for i in range(n_particles):
        if flag[i] == 1:
            acceleration_time[i] = 0
        else:
            acceleration_time = vel_max/qdd[i, :]

    for j in range(n_particles): # iterate through each joint
        max_velocity = vel_max[j]
        t_accel_r = acceleration_time[j]
        q0_r = q0[j]
        qf_r = qf[j] 
        tg = gen_traj(max_velocity, t_accel_r, q0_r, qf_r, tf, time)
        angle_array[j, :] = tg[0]
        velocity_array[j, :] = tg[1]
        accel_array[j, :] = tg[2]

    q_torq = np.transpose(angle_array)
    qd_torq = np.transpose(velocity_array)
    qdd_torq = np.transpose(accel_array)
    objective = calculate_energy(q_torq, qd_torq, qdd_torq, time)
    return objective

start1 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end1 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start2 = np.array([pi/2, -pi/2, pi/2, -pi/2, -pi/2, 0])
end2 = np.array([pi, -pi, 0, pi/4, -pi/2, pi])

start3 = np.array([0, 0, 0, 0, 0, 0])
end3 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])

start4 = np.array([pi, -pi/2, pi/2, -pi/2, -pi/2, 0])
end4 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start5 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end5 = np.array([2*pi/3, -pi/8, pi, -pi/2, 0, -pi/3])

trajectory_info = generate_traj_time(2, 201, start1, end1)
#plot_trajectory(trajectory_info)

# Definition of boundary condition below, see 08_Comparison.py
vel_max = np.zeros(6)
sign = np.zeros(6)
t_min = np.zeros(6) 
t_max = np.zeros(6) 
for j in range(6):
    q0 = trajectory_info[j].q[0]
    qf = trajectory_info[j].q[-1]
    if qf > q0:
        sign[j] = 1
    elif qf < q0:
        sign[j] = -1
    else:
        sign[j] = 0
    vel_max[j] = np.max(abs(trajectory_info[j].qd)) * sign[j]
    t_min[j] = abs(vel_max[j] / 100) 
    t_max[j] = abs(4/3 - vel_max[j] / 100) 
lower = np.zeros(6)
upper = np.zeros(6)
for i in range(6):
    if t_max[i] == 0:
        lower[i] = 0
    else:
        lower[i] = vel_max[i]/t_max[i]
    if t_min[i] == 0:
        upper[i] = 0
    else:
        upper[i] = vel_max[i]/t_min[i]
  
for j in range(6):
    if lower[j] < 0:
        temp = lower[j]
        lower[j] = upper[j]
        upper[j] = temp

lower = np.array([0, 0, 0, -10, -10, -10])
upper = np.array([80, 80, 0, 0, 0, 0])

bounds = (lower, upper)
start_time = ti.time()
num_particles = 100
num_iterations = 100
dim = 6
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9} # Coefficients for Particle Swarms
optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=6, bounds = bounds, options=options)
best_position, best_value = optimizer.optimize(energy_consumption, iters=num_iterations)
print(f'Optimization runtime: {ti.time() - start_time} seconds')
