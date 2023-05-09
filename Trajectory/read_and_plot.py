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

for j in range(6):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')


    ax1.plot(traj[6], traj[j].q)
    ax1.plot(t_evaluate, q_p[j, :])

    ax2.plot(traj[6], traj[j].qd)
    ax2.plot(t_evaluate, qd_p[j, :])

    ax3.plot(traj[6], traj[j].qdd)
    ax3.plot(t_evaluate, qdd_p[j, :])
    plt.show()
