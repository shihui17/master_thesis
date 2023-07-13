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

optimized_log = 'C:\Codes\master_thesis\Trajectory\Yu_logs\Log3.csv'
optimized_data = pd.read_csv(optimized_log)
time = optimized_data['time']
time = time[:-1]
q = []
qd = []
qdd = []

for i in range(6):
    q_read = np.array(optimized_data[f'q{i+1}'])
    q_read = q_read[:-1]
    q.append(q_read)
    qd_read = np.array(optimized_data[f'v{i+1}'])
    qd_read = qd_read[:-1]
    qd.append(qd_read)
    qdd_read = np.array(optimized_data[f'a{i+1}'])
    qdd_read = qdd_read[:-1]
    qdd.append(qdd_read)

q = np.array(q)
qd = np.array(qd)
qdd = np.array(qdd)
q_torq = np.transpose(q)
qd_torq = np.transpose(qd)
qdd_torq = np.transpose(qdd)
Log2 = 'C:\Codes\master_thesis\Trajectory\Yu_logs\Log2.csv'

data = pd.read_csv(Log2)
# Access columns using column names
time2 = data['time']
time2 = time2[:-1]
q2 = []
qd2 = []
qdd2 = []

for i in range(6):
    q_read2 = np.array(data[f'q{i+1}'])
    q_read2 = q_read2[:-1]
    q2.append(q_read2)
    qd_read2 = np.array(data[f'v{i+1}'])
    qd_read2 = qd_read2[:-1]
    qd2.append(qd_read2)
    qdd_read2 = np.array(data[f'a{i+1}'])
    qdd_read2 = qdd_read2[:-1]
    qdd2.append(qdd_read2)

q2 = np.array(q2)
qd2 = np.array(qd2)
qdd2 = np.array(qdd2)
q_torq2 = np.transpose(q2)
qd_torq2 = np.transpose(qd2)
qdd_torq2 = np.transpose(qdd2)
energy2 = calculate_energy(q_torq2, qd_torq2, qdd_torq2, time2)

start = np.array([-1.570796327,	-1.570796327,	1.570796327,	-1.570796327,	-1.570796327,	0])
end = np.array([1.215785543,	-1.319583654,	0.625792825,	-2.292949242,	-1.57085216,	-0.000166897])
t_alt = np.linspace(0, 2.45, 201)
tg = tools.trapezoidal(start[0], end[0], t_alt, V=1.45)
np.savetxt("original_time.txt", time2)
np.savetxt("original_vel.txt", qd2[0, :])
np.savetxt("1optimized_time1.txt", t_alt)
np.savetxt("1optimized_vel1.txt", tg.qd)
np.savetxt("2optimized_time2.txt", time)
np.savetxt("2optimized_vel2.txt", qd[0, :])
plt.plot(time2, qd2[0, :])
plt.plot(t_alt, tg.qd)
plt.plot(time, qd[0, :])
plt.show()

energy = calculate_energy(q_torq, qd_torq, qdd_torq, time)
print(energy)
print(energy2)