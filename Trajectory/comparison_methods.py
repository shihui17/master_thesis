from math import pi
import numpy as np
from roboticstoolbox import tools as tools
import matplotlib.pyplot as plt
from call_tau import *
from scipy.stats import truncnorm
from plot_traj import *
from traj import *
from HKA_kalman_gain import *
from energy_est import *
import time as ti

def gen_traj(vel_max, t_accel_rand, q0, qf, tf, time, V=None):
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

    if V is None:
        # if velocity not specified, compute it
        V = (qf - q0) / tf * 1.5
    else:
        V = abs(V) * np.sign(qf - q0)
        if abs(V) < (abs(qf - q0) / tf):
            raise ValueError("V too small")
        elif abs(V) > (2 * abs(qf - q0) / tf):
            raise ValueError("V too big")
    
    vel_max = V

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

start1 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end1 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])
trajectory_info = generate_traj_time(2, 201, start1, end1)
vel_max = np.zeros(6)
sign = np.zeros(6)
t_min = np.zeros(6) # initialize minimal acceleration time, will later be derived from boundary conditions
t_max = np.zeros(6) # initialize maximal acceleration time, will later be derived from boundary conditions
q0 = np.zeros(6)
qf = np.zeros(6)
time = trajectory_info[6]
tf = time[-1]
for j in range(6):
    q0[j] = trajectory_info[j].q[0]
    qf[j] = trajectory_info[j].q[-1]
    if qf[j] > q0[j]:
        sign[j] = 1
    elif qf[j] < q0[j]:
        sign[j] = -1
    else:
        sign[j] = 0
    vel_max[j] = np.max(abs(trajectory_info[j].qd)) * sign[j]

print(vel_max)
qdd_opt_sq = np.array([ 4.363e+00,  2.954e-01,  0.000e+00, -1.093e+00, -2.690e-01, -2.897e-01])
qdd_opt_nn = np.array([1.812e+01,  2.954e-01,  0.000e+00, -6.267e-01, -2.004e-01, -8.324e-01])
qdd_opt_trst = np.array([ 3.927e+00,  2.954e-01,  0.000e+00, -1.785e+00, -4.788e-01, -3.390e-01])
qdd_opt_ps = np.array([56.1978643,  46.40100772,    0, -0.22305548, -6.02647924, -3.02108289])
qdd_opt_hka = np.array([78.54545916,  0.31570746,  0.  ,       -1.70569077, -0.25708745, -0.19869935])
t = []
t1 = vel_max / qdd_opt_sq
t.append(t1)
t2 = vel_max / qdd_opt_nn
t.append(t2)
t3 = vel_max / qdd_opt_trst
t.append(t3)
t4 = vel_max / qdd_opt_ps
t.append(t4)
t5 = vel_max / qdd_opt_hka
t.append(t5)

q_mat = np.zeros((5, 6, 201))
qd_mat = np.zeros((5, 6, 201))
qdd_mat = np.zeros((5, 6, 201))
for i in range(5):
    for j in range(6): # iterate through each joint
        max_velocity = vel_max[j]
        t_accel_r = t[i][j]
        q0_r = q0[j]
        qf_r = qf[j] 
        tg = gen_traj(max_velocity, t_accel_r, q0_r, qf_r, tf, time)
        q_mat[i, j, :] = tg[0]
        qd_mat[i, j, :] = tg[1]
        qdd_mat[i, j, :] = tg[2]

for i in range(5): 
    plt.plot(time, qd_mat[i, 0, :])
plt.show()
np.savetxt("q_compare.txt", q_mat[:, 0, :])
np.savetxt("qd_compare.txt", qd_mat[:, 0, :])
np.savetxt("qdd_compare.txt", qdd_mat[:, 0, :])
np.savetxt("compare_time.txt", time)