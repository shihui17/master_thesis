import roboticstoolbox as rtb
import numpy as np
import math
from math import pi
from roboticstoolbox import tools as tools
import spatialmath as sm
import spatialgeometry as sg
import matplotlib as mpl
import matplotlib.pyplot as plt
from call_tau import *
from scipy.stats import truncnorm
from scipy.integrate import simpson
from plot_traj import *
from traj import *
from HKA_kalman_gain import *
from energy_est import *
import time as ti
from tau import *
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import csv

time = []
qd1 = []
qd2 = []
qd3 = []
qd4 = []
qd5 = []
qd6 = []
qdd1 = []
qdd2 = []
qdd3 = []
qdd4 = []
qdd5 = []
qdd6 = []
yu_log = 'C:\Codes\master_thesis\Trajectory\Yu_logs\Log2.csv'

with open(yu_log, 'r') as logging_data:
    reader = csv.reader(logging_data)
    next(reader)
    for row in reader:
        t = row[0]
        v1 = row[1]
        v2 = row[2]
        v3 = row[3]
        v4 = row[4]
        v5 = row[5]
        v6 = row[6]
        a1 = row[8]
        a2 = row[9]
        a3 = row[10]
        a4 = row[11]
        a5 = row[12]
        a6 = row[13]

        time.append(t)
        qd1.append(v1)
        qd2.append(v2)
        qd3.append(v3)
        qd4.append(v4)
        qd5.append(v5)
        qd6.append(v6)
        qdd1.append(a1)
        qdd2.append(a2)
        qdd3.append(a3)
        qdd4.append(a4)
        qdd5.append(a5)
        qdd6.append(a6)


time.pop()
qd1.pop()
qd2.pop()
qd3.pop()
qd4.pop()
qd5.pop()
qd6.pop()
qdd1.pop()
qdd2.pop()
qdd3.pop()
qdd4.pop()
qdd5.pop()
qdd6.pop()
time = np.array((time), dtype=float)
qd1 = np.array((qd1), dtype=float)
qd2 = np.array((qd2), dtype=float)
qd3 = np.array((qd3), dtype=float)
qd4= np.array((qd4), dtype=float)
qd5 = np.array((qd5), dtype=float)
qd6 = np.array((qd6), dtype=float)
qdd1 = np.array((qdd1), dtype=float)
qdd2 = np.array((qdd2), dtype=float)
qdd3 = np.array((qdd3), dtype=float)
qdd4= np.array((qdd4), dtype=float)
qdd5 = np.array((qdd5), dtype=float)
qdd6 = np.array((qdd6), dtype=float)

dx = np.diff(qdd1)
for i, value in enumerate(dx):
    if value < 0:
        turn1 = i
        break

qd_to_optimize = qd1[turn1 : len(qdd1)-turn1]
avg_qd = np.mean(qd_to_optimize)

# HKA definition:
start = np.array([-1.570796327,	-1.570796327,	1.570796327,	-1.570796327,	-1.570796327,	0])
end = np.array([1.215785543,	-1.319583654,	0.625792825,	-2.292949242,	-1.57085216,	-0.000166897])
hka_start = np.array([-1.570796327, 0, 0, 0, 0, 0])
hka_end = np.array([1.215785543, 0, 0, 0, 0, 0])
trajectory1 = tools.trapezoidal(start[0], end[0], time)
q1_hka = trajectory1.q
qd1_hka = trajectory1.qd
qdd1_hka = trajectory1.qdd
tf = time[-1]
t_min = 0.8*tf
t_max = 1.24*tf
mu_t = tf
iter = 0
N = 10
Nbest = 2
coeff = 0.6
q_mat = np.zeros((N, 6, 201)) # matrix to store randomly generated angular trajectories of all 6 joints, N trajectories in total, hence the size N x 6 x step
qd_mat = np.zeros((N, 6, 201)) # randomly generated velocity trajectory
qdd_mat = np.zeros((N, 6, 201))
sig_t = (t_max - t_min)/2
while iter <= 150:
    lb = (t_min - mu_t) / sig_t
    ub = (t_max - mu_t) / sig_t
    trunc_gen_t = truncnorm(lb, ub, loc=mu_t, scale=sig_t) 
    tf_rand = trunc_gen_t.rvs(size=N)
    energy_list_def = [('Energy','f8'), ('Number','i2')]
    energy_list = np.zeros((N), dtype = energy_list_def)

    for i in range(N):
        new_time = np.linspace(0, tf_rand[i], 201)
        new_traj = tools.trapezoidal(hka_start[0], hka_end[0], new_time)
        q_mat[i, 0, :] = new_traj.q
        qd_mat[i, 0, :] = new_traj.qd
        qdd_mat[i, 0, :] = new_traj.qdd

        q_torq = np.transpose(q_mat[i, :, :]) # transpose to be read by calculate_energy()
        qd_torq = np.transpose(qd_mat[i, :, :])
        qdd_torq = np.transpose(qdd_mat[i, :, :])
        energy_list[i] = (calculate_energy(q_torq, qd_torq, qdd_torq, new_time), i)

    sorted_energy_list = np.sort(energy_list, order='Energy') # sort energy consumption from lowest to highest
    num_array = sorted_energy_list['Number'] # the corresponding indices
    t_rand_index = num_array[0 : Nbest] # the indices of the Nbest acceleration time vectors
    post_t_rand = [tf_rand[i] for i in t_rand_index] # store accel time vectors into a big matrix to run through HKA
    #print(post_t_rand)
    mu_t_rand = np.mean(post_t_rand) # mean of Nbest candidates
    var_t_rand = np.var(post_t_rand) # variance of Nbest candidates
    new_mu_sig_t = kalman_gain(sig_t, var_t_rand, mu_t, mu_t_rand) # calculate Kalman gain, see HKA_kalman_gain.py
    mu_t = new_mu_sig_t[0] # new mean
    sig_t = new_mu_sig_t[1] # new std.dev.
    print(f'the diagonal of the covariance matrix:\n{var_t_rand}')

    if var_t_rand < 1e-5: # convergence criterion
        print(f'exited HKA at iter = {iter}')
        break

    iter = iter + 1

result_qd = qd_mat[num_array[0], :, :]
print(mu_t)
qd_max_optimized = np.max(result_qd[0, :])
print(qd_max_optimized)
print(avg_qd)
override_factor = np.max(result_qd[0, :])/avg_qd * coeff
print(override_factor)
