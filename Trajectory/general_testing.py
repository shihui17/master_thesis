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
"""
start_time1 = ti.time()
a = cal_tau(np.ones(6), np.ones(6), np.ones(6))
end_time1 = ti.time()
t1 = end_time1 - start_time1

start_time2 = ti.time()
b = calc_tau(np.ones(6), np.ones(6), np.ones(6))
end_time2 = ti.time()
t2 = end_time2 - start_time2

print(t1)
print(t2)
"""
Yu = rtb.models.DH.Yu()
q = np.array([0, -pi/2, 0, 0, 0, 0])
T_0e = Yu.fkine(q)
print(T_0e)