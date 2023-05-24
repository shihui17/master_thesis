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
start = np.array([-pi, -pi, pi/2, -pi/2, -pi/2, 0])
end = np.array([0, -0.749*pi, 0.69*pi, 0.444*pi, -0.8*pi, -pi])
traj = generate_traj_time(2, 201, start, end)
q_end = end
qd_end = np.zeros(6)
qdd_end = np.zeros(6)


xi = time_vec
q_p = np.zeros((6, 200))
qd_p = np.zeros((6, 200))
qdd_p = np.zeros((6, 200))


for joint_num in range(6):
    y = result_q[:, joint_num]
    yd = result_qd[:, joint_num]
    ydd = result_qdd[:, joint_num]
    yi = np.vstack((y, yd, ydd))
    yi = np.transpose(yi)
    #print(yi)

    print(len(xi))
    print(len(yi))
    func = BPoly.from_derivatives(xi, yi)
    t_evaluate = np.linspace(xi[0], xi[-1], num=200)
    q_p[joint_num, :] = func(t_evaluate)
    qd_p[joint_num, :] = func.derivative()(t_evaluate)
    qdd_p[joint_num, :] = func.derivative(2)(t_evaluate)

print(q_p)
save_dir = 'C:\Codes\master_thesis\Trajectory\Figures\Differentiation_tband/'

for j in range(6):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
    fig.suptitle(f'Trajecotry of joint {j+1}', fontsize=16)

    ax1.plot(traj[6], traj[j].q, label='Original angle')
    ax1.plot(t_evaluate, q_p[j, :], label='Optimized angle')
    ax1.set_xlabel('Travel time in s')
    ax1.set_ylabel('Joint angle in rad')
    ax1.legend()
    ax2.plot(traj[6], traj[j].qd, label='Original velocity')
    ax2.plot(t_evaluate, qd_p[j, :], label='Optimized velocity')
    ax2.set_xlabel('Travel time in s')
    ax2.set_ylabel('Joint velocity in rad/s')
    ax2.legend()
    ax3.plot(traj[6], traj[j].qdd, label='Original acceleration')
    ax3.plot(t_evaluate, qdd_p[j, :], label='Optimized acceleration')
    ax3.set_xlabel('Travel time in s')
    ax3.set_ylabel('Joint acceleration in rad/s^2')
    ax3.legend()
    plt.show()
    fig.savefig(save_dir+f'diff_joint{j+1}.png')