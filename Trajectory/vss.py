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
from scipy.interpolate import BPoly, CubicHermiteSpline
from scipy.integrate import simpson

result_q = np.loadtxt("result_q.txt")
result_qd = np.loadtxt("result_qd.txt")
result_qdd = np.loadtxt("result_qdd.txt")
time_vec = np.loadtxt("time_vec.txt")

x = time_vec
y = result_q[:, 0]
dydx = result_qd[:, 0]

func = CubicHermiteSpline(x, y, dydx)
t_eval = np.linspace(time_vec[0], time_vec[-1], num=201)
y_eval = func(t_eval)
yd_eval = func.derivative(1)(t_eval)
ydd_eval = func.derivative(2)(t_eval)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="constrained")
ax1.plot(t_eval, y_eval)
ax2.plot(t_eval, yd_eval)
ax3.plot(t_eval, ydd_eval)
plt.show()