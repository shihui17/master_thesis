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
from scipy.interpolate import CubicHermiteSpline, BPoly, interp1d
from traj import *

# same goes for the lower bound
# to generalise, define the following function:

joint_data = generate_traj_time(2)
angle = joint_data[1].q
velocity = joint_data[1].qd
accel = joint_data[1].qdd
time = joint_data[6]

def create_tolerance_bands(angle, velocity, time, width, upper_lower):
    allowed = ["upper", "lower"]
    t0 = 0 # start
    tf = time[-1] # finish
    q0 = angle[0]
    qf = angle[-1]
    V = (qf - q0) / tf * 1.5
    t_accel = np.round((q0 - qf + V * tf) / V, 2) # end of acceleration, rounded to 2 decimals to exactly match the time points in joint_data[6]
    t_brake = np.round(tf - t_accel, 2) # start of braking
    i_accel = np.where(time == t_accel)[0][0]
    i_brake = np.where(time == t_brake)[0][0]

    t0 = t0
    t1 = t_accel
    qd0 = 0

    if upper_lower in allowed:
        if upper_lower == "lower":
            q1 = angle[0] + (angle[i_accel] - angle[0]) * width # generate a new angle for lower bound
            print(f'Test q1 lower = {q1}')
        elif upper_lower == "upper":
            q_l = angle[0] + (angle[i_accel] - angle[0]) * width # angle for lower bound
            q1 = 2 * angle[i_accel] - q_l # flip it around angle[i_accel] to get upper bound
            print(f'Test q1 upper = {q1}')
    else:
        raise ValueError(f"Invalid string passed in upper_lower")
    qd1 = velocity[i_accel] # assume unchanged velocity at t_accel
            #print(velocity)
            #print(angle)

    x = np.array([t0, t1])
    y = np.array([q0, q1])
    dydx = np.array([qd0, qd1])

    cubic_spline = CubicHermiteSpline(x, y, dydx)
    t_sample = np.linspace(t0, t1, int(100*(t1-t0)+1))
    q_spline = cubic_spline(t_sample)

    # so far so good, now the second phase (t_accel to t_brake)

    t0 = t_accel
    t1 = t_brake
    x = np.array([t0, t1])
    q0 = q1
    y = np.array([q0, q1])
    qd0 = qd1
 
    def linear_k(k, x1, y1, x):
        return k * (x - x1) + y1

    t_sample2 = np.linspace(t0, t1, int(100*(t1-t0)+1))
    q_spline2 = [linear_k(qd0, t0, q0, t) for t in t_sample2]

    # braking

    t0 = t_brake
    t1 = tf
    qd0 = qd1
    q0 = q_spline2[-1]
    q1 = qf
    qd1 = 0
    #print(velocity)
    #print(angle)

    x = np.array([t0, t1])
    y = np.array([q0, q1])
    dydx = np.array([qd0, qd1])

    cubic_spline2 = CubicHermiteSpline(x, y, dydx)
    t_sample3 = np.linspace(t0, t1, int(100*(t1-t0)+1))
    q_spline3 = cubic_spline2(t_sample3)

    t_total = np.concatenate((t_sample, t_sample2, t_sample3))
    angle_bound = np.concatenate((q_spline, q_spline2, q_spline3))

    return t_total, angle_bound

"""
results1 = create_tolerance_bands(angle, velocity, time, 0.5)
t_total = results1[0]
angle_upper = results1[1]

results2 = create_tolerance_bands(angle, velocity, time, -0.5)
t_total2 = results2[0]
angle_lower = results2[1]


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(t_total, angle_upper, label='cubic', linestyle='dashed', color='green')
ax1.plot(time, angle, label='original')
ax1.plot(t_total2, angle_lower, label='cubic', linestyle='dashed', color='green')

ax1.legend()
plt.show()
"""