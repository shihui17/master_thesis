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

def kalman_gain(sig, var_post, mu, mu_meas):
    L = np.divide(np.square(sig), (np.square(sig) + var_post))
    mu_post = mu + np.multiply(L, (mu_meas - mu))
    P = np.square(sig) - np.multiply(L, np.square(sig))
    s = np.min([1, np.mean(np.sqrt(var_post))])
    a = 0.6 * (s**2 / (s**2 + np.max(P)))
    #print(P)
    #print(var_q_post)
    sig_new = sig + a * (np.sqrt(P) - sig)
    mu_new = mu_post
    return mu_new, sig_new

def HKA(N, Nbest, D, alpha, sd, n):
    mu_q = np.zeros(n)
    sig_q = np.zeros(n)
    mu_qd = np.zeros(n)
    sig_qd = np.zeros(n)
    mu_qdd = np.zeros(n)
    sig_qdd = np.zeros(n)
    result = np.zeros(n+1)
    iter = 0
    max_iter = 150
    lb = matlib.repmat(D[0, :], N, 1)
    ub = matlib.repmat(D[1, :], N, 1)
    assble_q = np.zeros((N, n))
    assble_qd = np.zeros((N, n))
    assble_qdd = np.zeros((N, n))
    joint = generate_traj(20)

    # initialize mean vector mu_q, mu_qd, mu_qdd, and std vector sig_q, sig_qd, sig_qdd
    for i in range(n):
        mu_q[i] = joint[i].q[10]
        #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
        sig_q[i] = 0.01 # initialize std vector
        mu_qd[i] = joint[i].qd[10]
        #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
        sig_qd[i] = 0.01 # initialize std vector
        mu_qdd[i] = joint[i].qdd[10]
        #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
        sig_qdd[i] = 0.01 # initialize std vector

    print(mu_q)
    print(mu_qd)
    print(mu_qdd)

    while iter <= max_iter:
        for i in range (n):
            s_q = np.random.normal(mu_q[i], sig_q[i], N)
            assble_q[:, i] = np.transpose(s_q)
            s_qd = np.random.normal(mu_qd[i], sig_qd[i], N)
            assble_qd[:, i] = np.transpose(s_qd)
            s_qdd = np.random.normal(mu_qdd[i], sig_qdd[i], N)
            assble_qdd[:, i] = np.transpose(s_qdd)
        ref_qd = assble_qd
        #torque = cal_tau(assble_q[0, :], assble_qd[0, :], assble_qdd[0, :])
        #print(assble_q)
        #print(torque)
        power = np.zeros((N, 6))
        val_func_def = [('Power','f8'), ('Number','i2')]
        val_func = np.zeros((N), dtype = val_func_def)
        for i in range(N):
            torq_vec = np.abs(cal_tau(assble_q[i, :], assble_qd[i, :], assble_qdd[i, :]))
            velocity_vec = assble_qd[i, :]
            power[i, :] = np.abs(np.multiply(torq_vec, velocity_vec))
            val_func[i] = (np.linalg.norm(power[i, :], 1), i+1) 
        #print(f'The power output for each joint with {N} samples is :\n{power}\n')
        #print(f'The total power output of the robot (col.1) with the corresponding sample number (col.2) is:\n{val_func}\n')
        #print(f'The assembled joint angle matrix for all 6 joints with {N} samples is:\n {assble_q}\n')
        #print(f'The assembled joint velocity matrix for all 6 joints with {N} samples is:\n {assble_qd}\n')
        #print(f'The assembled joint acceleration matrix for all 6 joints with {N} samples is:\n {assble_qdd}\n')

        #print(val_func)
        sort_val_func = np.sort(val_func, order='Power')
        #print(sort_val_func)

        post_q = np.zeros((10, 6))
        post_qd = np.zeros((10, 6))
        post_qdd = np.zeros((10, 6))
        num_array = sort_val_func['Number']
        #print(num_array[0])
        #print(assble_q)
        #print(post_q)

        for i in range(10):
            num = num_array[i]
            post_q[i, :] = assble_q[num-1, :]
            post_qd[i, :] = assble_qd[num-1, :]
            post_qdd[i, :] = assble_qdd[num-1, :]

        mu_q_meas = np.mean(post_q, 0)
        mu_qd_meas = np.mean(post_qd, 0)
        mu_qdd_meas = np.mean(post_qdd, 0)
        var_q_post = np.var(post_q, 0)
        var_qd_post = np.var(post_qd, 0)
        var_qdd_post = np.var(post_qdd, 0)
        #print(mu_q_post)
        #print(mu_qd_post)
        #print(mu_qdd_post)
        
        #Compute Kalman gain for q, qd, qdd
        new_mu_sig_q = kalman_gain(sig_q, var_q_post, mu_q, mu_q_meas)
        mu_q_old = mu_q
        mu_q = new_mu_sig_q[0]
        sig_q = new_mu_sig_q[1]

        new_mu_sig_qd = kalman_gain(sig_qd, var_qd_post, mu_qd, mu_qd_meas)
        mu_qd_old = mu_qd
        mu_qd = new_mu_sig_qd[0]
        sig_qd = new_mu_sig_qd[1]

        new_mu_sig_qdd = kalman_gain(sig_qdd, var_qdd_post, mu_qdd, mu_qdd_meas)
        mu_qdd_old = mu_qdd
        mu_qdd = new_mu_sig_qdd[0]
        sig_qdd = new_mu_sig_qdd[1]

        #print(f'old mean vector for q:\n{mu_q_old}\nfor qd:\n{mu_qd_old}\n:for qdd:\n{mu_qdd_old}\n')
        """
        L = np.divide(np.square(sig_q), (np.square(sig_q) + var_q_post))
        mu_q_post = mu_q + np.multiply(L, (mu_q_meas - mu_q))
        P = np.square(sig_q) - np.multiply(L, np.square(sig_q))
        #print(P)
        #print(var_q_post)
        sig_q = sig_q + (np.sqrt(P) - sig_q)
        mu_q = mu_q_post
        """
        #print(assble_qd)

        iter = iter + 1
    print(f'Power output of each joint converges at {assble_qd[0, :]} rad/s, total power output: {sort_val_func[0]}')
    print(assble_q[0, :], assble_qd[0, :], assble_qdd[0, :])




HKA(20, 0, np.array([[0, 0, 0, 0, 0, 0], [0.0003, 5, 6, 8, 20, 10]]), 0, 0, 6)

