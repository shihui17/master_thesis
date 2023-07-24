'''
@author: Shihui Liu

A comparison between HKA and other optimization methods provided by scipy.optimize. Arguments that can be given to scipy.optimize() are: 'Trust-constr', 'SLSQP', 'Nelder-Mead'. 
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

def gen_traj(vel_max, t_accel_rand, q0, qf, tf, time):
    """
    Generate a trapezoidal trajectory in joint space given parameters generated by HKA
    :param vel_max: maximal (allowed) angular velocity of the robot joint
    :param t_accel_rand: the acceleration time, randomly generated by HKA
    :param q0: initial joint angle at t = 0
    :param qf: end joint angle at t = tf
    :param tf: trajectory time
    :param time: the discretized time array, with time[0] = 0, time[-1] = tf
    :return q: array of joint angle with the size of len(time)
    :return qd: array of joint velocity with the size of len(time)
    :return qdd: array of joint acceleration with the size of len(time)
    """
    q = []
    qd = []
    qdd = []

    if q0 == qf: # if the joint is stationary
        return (np.full(len(time), q0), np.zeros(len(time)), np.zeros(len(time))) # return q = q0, qd = 0, and qdd = 0 for the entire trajectory time
    else:
        a_accel_rand = (vel_max / t_accel_rand) # random acceleration, dependent on t_accel_rand generated by HKA
        t_brake_rand = 2 * (qf - q0) / vel_max + t_accel_rand - tf # the corresponding brake time
        a_brake_rand = vel_max / (tf - t_brake_rand) # the corresponding decceleration

        for tk in time: # the following trajectory planning is formulated according to the tools.trapezoidal method in roboticstoolbox

            if tk < 0:
                qk = q0
                qdk = 0
                qddk = 0
            elif tk <= t_accel_rand:
                qk = q0 + 0.5 * a_accel_rand * tk**2
                qdk = a_accel_rand * tk
                qddk = a_accel_rand
            elif tk <= t_brake_rand:
                qk = q0 + 0.5 * a_accel_rand * t_accel_rand**2 + vel_max * (tk - t_accel_rand)
                qk = vel_max * tk + q0 + 0.5 * a_accel_rand * t_accel_rand**2 - vel_max * t_accel_rand
                qdk = vel_max
                qddk = 0
            elif tk <= tf:
                qk = q0 + 0.5 * a_accel_rand * t_accel_rand**2 + vel_max * (t_brake_rand - t_accel_rand) + (vel_max * (tk - t_brake_rand) - 0.5 * a_brake_rand * (tk - t_brake_rand)**2)
                qdk = vel_max - a_brake_rand * (tk - t_brake_rand)
                qddk = -a_brake_rand
            else:
                qk = qf
                qdk = 0
                qddk = 0

            q.append(qk)
            qd.append(qdk)
            qdd.append(qddk)

        return (np.array(q), np.array(qd), np.array(qdd))

def energy_consumption(qdd): # this is the objective function to be optimized by scipy.optimize. Procedures very similar to 03_Profile_Optimization.py (almost identical)
    q0 = np.zeros(6)
    qf = np.zeros(6)
    step = 201
    angle = np.zeros((6, step)) 
    velocity = np.zeros((6, step))
    accel = np.zeros((6, step))
    vel_max = np.zeros(6)
    t_accel = np.zeros(6) 
    t_brake = np.zeros(6)
    angle_array = np.zeros((6, 201))
    velocity_array = np.zeros((6, 201))
    accel_array = np.zeros((6, 201))
    acceleration_time = np.zeros(6)
    flag = np.zeros(6)
    t_min = np.zeros(6) # initialize minimal acceleration time, will later be derived from boundary conditions
    t_max = np.zeros(6) # initialize maximal acceleration time, will later be derived from boundary conditions
    a = np.zeros(6)
    start1 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
    end1 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])
    trajectory_info = generate_traj_time(2, step, start1, end1)
    time = trajectory_info[6]
    t0 = 0 # start
    tf = time[-1] # finish
    for j in range(6): # reads necessary data from the original trajectory
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
            #print('test flag')
        else: 
            vel_max[j] = (qf[j] - q0[j]) / tf * 1.5 
            t_accel[j] = np.round((q0[j] - qf[j] + vel_max[j] * tf) / vel_max[j], 2) # end of acceleration, rounded to 2 decimals to exactly match the time points in traj[6]
            t_brake[j] = np.round(tf - t_accel[j], 2) # start of braking
            a[j] = vel_max[j] / t_accel[j]
            t_min[j] = abs(vel_max[j] / 100) # boundary condition: maximal angular acceleration 250 s^-2, set to 100 s^-2 to avoid drastic joint movement
            t_max[j] = abs(tf - tf / 3 - vel_max[j] / 100) # maximal allowed acceleration time is reached when the joint has to brake with the maximal angular acceleration in order to reach end configuration (100 s^-2)
    #print(f'flag is {flag}')
    for i in range(6):
        if flag[i] == 1:
            acceleration_time[i] = 0
        else:
            acceleration_time[i] = vel_max[i]/qdd[i]
            #print(f'{qdd[i]}, {i}')

    for j in range(6): # iterate through each joint
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
plot_trajectory(trajectory_info)
vel_max = np.zeros(6)
sign = np.zeros(6)
t_min = np.zeros(6) # initialize minimal acceleration time, will later be derived from boundary conditions
t_max = np.zeros(6) # initialize maximal acceleration time, will later be derived from boundary conditions

# The section of code below is to define the boundaries of the optimization problem, also almost identical to 03_Profile_Optimization.py
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
    t_min[j] = abs(vel_max[j] / 100) # boundary condition: maximal angular acceleration 100 s^-2
    t_max[j] = abs(4/3 - vel_max[j] / 100) # maximal allowed acceleration time is reached when the joint has to brake with the maximal angular acceleration in order to reach end configuration (100 s^-2)

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
#print(lower)
#print(upper)
b1 = (lower[0], upper[0])
b2 = (lower[1], upper[1])
b3 = (lower[2], upper[2])
b4 = (lower[3], upper[3])
b5 = (lower[4], upper[4])
b6 = (lower[5], upper[5])

bnds = (b1, b2, b3, b4, b5, b6)
qdd0 = np.array([trajectory_info[0].qdd[0], trajectory_info[1].qdd[0], trajectory_info[2].qdd[0], trajectory_info[3].qdd[0], trajectory_info[4].qdd[0], trajectory_info[5].qdd[0]])
#qdd0 = np.array([79.02070876,  0.31083316,  0.       ,  -1.16697786, -0.51799572, -0.41148557])

start_time = ti.time() # timer that records algorithm runtime
sol = minimize(energy_consumption, qdd0, method='Trust-constr', bounds=bnds)
print(f'Optimization runtime: {ti.time() - start_time} seconds')
print(sol)

