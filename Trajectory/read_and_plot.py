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

result_q = np.loadtxt("prof_result_q.txt")
result_qd = np.loadtxt("prof_result_qd.txt")
result_qdd = np.loadtxt("prof_result_qdd.txt")
time_vec = np.loadtxt("original_time.txt")

print(np.shape(result_q))
print(np.shape(time_vec))

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

joint_data = generate_traj_time(2, 201, start1, end1)
for j in range(6):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(time_vec, result_q[j, :], label=f'optimized traj for joint {j+1}', color='green')
    ax1.plot(joint_data[6], joint_data[j].q, label=f'orignal traj for joint {j+1}', color='red')
    ax1.set_xlabel('Travel time in s')
    ax1.set_ylabel('Joint angle in rad')
    ax1.legend()
    ax2.plot(time_vec, result_qd[j, :], label=f'optimized traj for joint {j+1}', color='g')
    ax2.plot(joint_data[6], joint_data[j].qd, label=f'orignal traj for joint {j+1}', color='r')
    ax2.set_xlabel('Travel time in s')
    ax2.set_ylabel('Joint velocity in rad/s')
    ax3.plot(time_vec, result_qdd[j, :], label=f'optimized traj for joint {j+1}', color='g')
    ax3.plot(joint_data[6], joint_data[j].qdd, label=f'orignal traj for joint {j+1}', color='r')
    ax3.set_xlabel('Travel time in s')
    ax3.set_ylabel('Joint acceleration in rad/s^2')
    fig.suptitle(f"Trajectory for Joint {j+1}", fontsize=16)
    
    #fig.savefig(f'C:\Codes\master_thesis\Trajectory\Figures\Profile_optimization/prof_opt_joint{j+1}.png')
    plt.show()

q_torque = np.transpose(result_q)
qd_torque = np.transpose(result_qd)
qdd_torque = np.transpose(result_qdd)
power_list = []
for i in range(q_torque.shape[0]):
    torq_vec = cal_tau(q_torque[i, :], qd_torque[i, :], qdd_torque[i, :])
    velocity_vec = qd_torque[i, :]
    power = abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1))
    power_list.append(power)

power = np.array(power_list)
np.savetxt("power.txt", power)
plt.plot(time_vec, power_list)
plt.show()