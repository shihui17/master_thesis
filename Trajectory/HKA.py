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

def HKA(N, Nbest, D, alpha, sd, n):
    mu = np.zeros(n)
    sig = np.zeros(n)
    result = np.zeros(n+1)
    max_iter = 300
    lb = matlib.repmat(D[0, :], N, 1)
    ub = matlib.repmat(D[1, :], N, 1)
    joint = generate_traj(20)

    for i_q in range(n):
        mu[i_q] = joint[i_q].qd[10]
        #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
        sig[i_q] = 0.02 # initialize std vector
    assble_q = np.zeros((N, n))
    for ctr_q in range (n):
        s = np.random.normal(mu[ctr_q], sig[ctr_q], N)
        assble_q[:, ctr_q] = np.transpose(s)

    for i_qd in range(n):
        mu[i_qd] = joint[i_qd].qd[10]
        #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
        sig[i_qd] = 0.02# initialize std vector
    assble_qd = np.zeros((N, n))
    for ctr_qd in range (n):
        s = np.random.normal(mu[ctr_qd], sig[ctr_qd], N)
        assble_qd[:, ctr_qd] = np.transpose(s)

    for i_qdd in range(n):
        mu[i_qdd] = joint[i_qdd].qdd[10]
        #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
        sig[i_qdd] = 0.02 # initialize std vector
    assble_qdd = np.zeros((N, n))
    for ctr_qdd in range (n):
        s = np.random.normal(mu[ctr_qdd], sig[ctr_qdd], N)
        assble_qdd[:, ctr_qdd] = np.transpose(s)

    #torque = cal_tau(assble_q[0, :], assble_qd[0, :], assble_qdd[0, :])
    #print(assble_q)
    #print(torque)
    power = np.zeros((N, 6))
    val_func = np.zeros([N, 2])
    for i in range(N):
        torq_vec = np.abs(cal_tau(assble_q[i, :], assble_qd[i, :], assble_qdd[i, :]))
        velocity_vec = assble_qd[i, :]
        power[i, :] = np.abs(np.multiply(torq_vec, velocity_vec))
        val_func[i, :] = np.array([np.linalg.norm(power[i, :], 1), i+1])
    print(val_func)
    print(assble_q)

    #tomorrow: associate the cost function values with the corresponding vectors that yield the values

    



    



    
    iter = 1
    conv_factor = 50000

HKA(20, 0, np.array([[0, 0, 0, 0, 0, 0], [0.0003, 5, 6, 8, 20, 10]]), 0, 0, 6)

    