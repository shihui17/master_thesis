import roboticstoolbox as rtb
import numpy as np
import math
from math import pi
from roboticstoolbox import tools as tools
import matplotlib.pyplot as plt
from numpy import matlib
from call_tau import *
from traj import *
from tolerance import *
from lerp import *
from scipy.stats import truncnorm
from scipy.interpolate import BPoly
from scipy.integrate import simpson

result_q = np.loadtxt("result_q.txt")
result_qd = np.loadtxt("result_qd.txt")
result_qdd = np.loadtxt("result_qdd.txt")
time_vec = np.loadtxt("time_vec.txt")

def calculate_power(result_q, result_qd, result_qdd, time_vec):
    q_sample = np.zeros((6, 500))
    qd_sample = np.zeros((6, 500))
    qdd_sample = np.zeros((6, 500))
    t1 = time_vec[0]
    t2 = time_vec[-1]
    xi = time_vec

    for joint_num in range(6):
        y = result_q[:, joint_num]
        yd = result_qd[:, joint_num]
        ydd = result_qdd[:, joint_num]
        yi = np.vstack((y, yd, ydd))
        yi = np.transpose(yi)
        #print(yi)

        func = BPoly.from_derivatives(xi, yi)
        t_evaluate = np.linspace(xi[0], xi[-1], num=500)
        q_sample[joint_num, :] = func(t_evaluate)
        qd_sample[joint_num, :] = func.derivative()(t_evaluate)
        qdd_sample[joint_num, :] = func.derivative(2)(t_evaluate)

    q_calc = np.transpose(q_sample)
    qd_calc = np.transpose(qd_sample)
    qdd_calc = np.transpose(qdd_sample)
    power_val = []

    for k in range(500):

        torq_vec = abs(cal_tau(q_calc[k, :], qd_calc[k, :], qdd_calc[k, :]))
        velocity_vec = abs(qd_calc[k, :])
        #if 190 <= k <= 210:
            #print(torq_vec)
            #print(velocity_vec)
            #print(abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1)))

        power_val.append(abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1))) # element-wise multiplication, then take 1-Norm

    return power_val, t_evaluate, q_sample, qd_sample, qdd_sample

results = calculate_power(result_q, result_qd, result_qdd, time_vec)
power_val = results[0]
t_val = results[1]
q_graph = results[2]
qd_graph = results[3]
qdd_graph = results[4]

og_result = generate_traj_time(2)

q_og = np.zeros((6, 201))
qd_og = np.zeros((6, 201))
qdd_og = np.zeros((6, 201))
time_og = og_result[6]

for joint_num in range(6):

    q_og[joint_num, :] = og_result[joint_num].q
    qd_og[joint_num, :] = og_result[joint_num].qd
    qdd_og[joint_num, :] = og_result[joint_num].qdd

q_og_calc = np.transpose(q_og)
qd_og_calc = np.transpose(qd_og)
qdd_og_calc = np.transpose(qdd_og)
power_og_val = []

for k in range(201):

    torq_vec = np.absolute(cal_tau(q_og_calc[k, :], qd_og_calc[k, :], qdd_og_calc[k, :]))
    velocity_vec = np.absolute(qd_og_calc[k, :])
    power_og_val.append(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1)) # element-wise multiplication, then take 1-Norm


energy_og = simpson(power_og_val, time_og)
print(energy_og)
energy_opt = simpson(power_val, t_val)
print(energy_opt)

for j in range(6):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="constrained")
    ax1.plot(t_val, q_graph[j, :])
    ax1.plot(time_og, q_og[j, :])

    ax2.plot(t_val, qd_graph[j, :])
    ax2.plot(time_og, qd_og[j, :])

    ax3.plot(t_val, qdd_graph[j, :])
    ax3.plot(time_og, qdd_og[j, :])



plt.plot(t_val, power_val)
plt.plot(time_og, power_og_val)
plt.show()


