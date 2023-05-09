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

result_q = np.loadtxt("result_q_int.txt")
result_qd = np.loadtxt("result_qd_int.txt")
result_qdd = np.loadtxt("result_qdd_int.txt")
time_vec = np.loadtxt("time_vec_int.txt")
traj = generate_traj_time(2, 201)
q_end = np.array([pi, -0.749*pi, 0.689*pi, 0.444*pi, -0.8*pi, -pi])
qd_end = np.zeros(6)
qdd_end = np.zeros(6)
xi = time_vec[:-1]
q_p = np.zeros((6, 201))
qd_p = np.zeros((6, 201))
qdd_p = np.zeros((6, 201))


for joint_num in range(6):
    y = result_q[:, joint_num]
    yd = result_qd[:, joint_num]
    ydd = result_qdd[:, joint_num]
    yi = np.vstack((y, yd, ydd))
    yi = np.transpose(yi)
    #print(yi)

    #print(len(xi))
    #print(len(yi))
    func = BPoly.from_derivatives(xi, yi)
    t_evaluate = np.linspace(xi[0], xi[-1], num=201)
    q_p[joint_num, :] = func(t_evaluate)
    qd_p[joint_num, :] = func.derivative()(t_evaluate)
    qdd_p[joint_num, :] = func.derivative(2)(t_evaluate)

q = np.loadtxt("result_q_int.txt")
qd = np.loadtxt("result_qd_int.txt")
qdd = np.loadtxt("result_qdd_int.txt")
time = np.loadtxt("time_vec_int.txt")
time[-1] = 2
joint = generate_traj_time(2, 201)

qdd_plot = []
t_list = t_evaluate

def poly4_interp(x):
    A_matrix = np.array([[x**4, x**3, x**2, x, 1], [4*x**3, 3*x**2, 2*x, 1, 0], [12*x**2, 6*x, 2, 0, 0]])
    return A_matrix

def poly4(a, b, c, d, e, x):
    func = a * x**4 + b * x**3 + c * x**2 + d * x + e
    return func

def poly4_d(a, b, c, d, x):
    func = 4 * a * x**3 + 3* b * x**2 + 2 * c * x + d
    return func

def poly4_dd(a, b, c, x):
    func = 12 * a * x**2 + 6* b * x + 2 * c
    return func


A_matrix1 = poly4_interp(time[-2])
print(time[-2])
print(A_matrix1)
A_matrix2 = poly4_interp(time[-1])
A_matrix2 = np.delete(A_matrix2, 2, 0)
print(A_matrix2)
A_matrix_whole = np.vstack((A_matrix1, A_matrix2))
print(A_matrix_whole)
tt = np.linspace(time[-2], time[-1], num=100)
q_j1 = []
qd_j1 = []
qdd_j1 = []

for joint_num in range(6):
    B_vec = np.array([q[-1, joint_num], qd[-1, joint_num], qdd[-1, joint_num], q_end[joint_num], 0])
#print(B_vec)
    sol = np.linalg.solve(A_matrix_whole, B_vec)
    yt = [poly4(sol[0], sol[1], sol[2], sol[3], sol[4], t) for t in tt]
    yt_d = [poly4_d(sol[0], sol[1], sol[2], sol[3], t) for t in tt]
    yt_dd = [poly4_dd(sol[0], sol[1], sol[2], t) for t in tt]
    qdd_end[joint_num] = yt_dd[-1]
    t_j1 = np.concatenate((t_list, tt))
    q_j1.append(np.concatenate((q_p[joint_num, :], yt)))
    qd_j1.append(np.concatenate((qd_p[joint_num, :], yt_d)))
    qdd_j1.append(np.concatenate((qdd_p[joint_num, :], yt_dd)))



q_total = np.vstack((q, q_end))
qd_total = np.vstack((qd, qd_end))
qdd_total = np.vstack((qdd, qdd_end))
save_dir = 'C:/Codes/master_thesis/Trajectory/Figures/integration_joint_traj/'

for joint_num in range(6):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')

    fig.suptitle(f'Trajecotry of joint {joint_num+1}', fontsize=16)
    ax1.plot(joint[6], joint[joint_num].q, color='blue', label='Original angle')
    ax1.plot(time, q_total[:, joint_num], 'r+', label='Optimized angle')
    ax1.plot(t_j1, q_j1[joint_num], color='green', label='Optimized angle profile')
    ax1.set_xlabel('Travel time in s')
    ax1.set_ylabel('Joint angle in rad')
    ax1.legend()

    ax2.plot(joint[6], joint[joint_num].qd, color='blue', label='Original velocity')
    ax2.plot(time, qd_total[:, joint_num], 'r+', label='Optimized velocity')
    ax2.plot(t_j1, qd_j1[joint_num], color='green', label='Optimized velocity profile')
    ax2.set_xlabel('Travel time in s')
    ax2.set_ylabel('Joint velocity in rad/s')
    ax2.legend()

    ax3.plot(joint[6], joint[joint_num].qdd, color='blue', label='Original acceleration')
    ax3.plot(time, qdd_total[:, joint_num], 'r+', label='Optimized acceleration')
    ax3.plot(t_j1, qdd_j1[joint_num], color='green', label='Optimized accel. profile')
    ax3.set_xlabel('Travel time in s')
    ax3.set_ylabel('Joint acceleration in rad/s^2')
    ax3.legend()

    plt.show()
    fig.savefig(save_dir + f'figure_joint{joint_num+1}.png')