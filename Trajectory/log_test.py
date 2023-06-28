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
"""
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
yu_log = 'C:\Codes\master_thesis\Trajectory\Yu_logs\Log3.csv'

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
"""
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