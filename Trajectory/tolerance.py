'''
@author: Shihui Liu

Use create_tolerance_bands() to generate tolerance band around original trajectories. This function is built in all HKA optimizers that utilize a tolerance band and does not need to be further modified.
'''

import numpy as np
from math import pi
from roboticstoolbox import tools as tools
from call_tau import *
from scipy.interpolate import CubicHermiteSpline, BPoly, interp1d
from traj import *
import matplotlib.pyplot as plt

def create_tolerance_bands(angle, velocity, time, width, upper_lower):
    """
    Generates tolerance band around original angular trajectory using cubic Hermite splines
    :param angle: original angle for all joints
    :param velocity: original velocity for all joints
    :param time: original trajectory time
    :param width: controls the width of the tolerance band, ranging from 0 to 1
    :param upper_lower: string parameter. Pass "upper" to function to generate upper boundary of the band, "lower" for lower boundary
    """
    allowed = ["upper", "lower"]
    t0 = 0 # start of trajectory
    tf = time[-1] # end of trajectory
    q0 = angle[0] # initial angle
    qf = angle[-1] # end angle
    if q0 == qf:
        return False

    V = (qf - q0) / tf * 1.5 # according to tools.trapezoidal in roboticstoolbox
    t_accel = np.round((q0 - qf + V * tf) / V, 2) # end of acceleration, rounded to 2 decimals to exactly match the time points in joint_data[6]
    t_brake = np.round(tf - t_accel, 2) # start of braking
    i_accel = np.where(time == t_accel)[0][0] # find the time index where acceleration begins

    t1 = t_accel
    qd0 = 0 # initial velocity is always 0

    if upper_lower in allowed:
        q11 = angle[0] + (angle[i_accel] - angle[0]) * width # lower boundary
        q12 = 2 * angle[i_accel] - q11 # upper boundary
        if upper_lower == "lower":
            if angle[i_accel] > angle[0]: # if velocity is positive (joint angle at the end of acceleration is greater than joint angle at t = 0)
                q1 = q11 
            else: # if velocity is negative
                q1 = q12 # the upper boundary for when velocity is positive becomes the lower boundary in this case
        elif upper_lower == "upper":
            if angle[i_accel] > angle[0]:
                q1 = q12 
            else:
                q1 = q11
    else:
        raise ValueError(f"Invalid string passed in upper_lower")
    qd1 = velocity[i_accel] # after acceleration, joint moves with original maximal velocity, this way the middle section (the straight line) of the tolerance band stays parallel to the original trajectory

    x = np.array([t0, t1]) # x-coordinates (time) of control points for cubic Hermite interpolation
    y = np.array([q0, q1]) # y-coordinates (angle) of control points
    dydx = np.array([qd0, qd1]) # derivatives at control points (velocity)
    cubic_spline = CubicHermiteSpline(x, y, dydx)
    t_sample = np.linspace(t0, t1, int(100*(t1-t0)+1)) # sample time array
    q_spline = cubic_spline(t_sample) # generate cubic spline

    # below is the middle section, generated by linear interpolation 
    t0 = t_accel # starting point is end of acceleration
    t1 = t_brake # end point is start of braking
    x = np.array([t0, t1])
    q0 = q1
    y = np.array([q0, q1])
    qd0 = qd1
 
    def linear_k(k, x1, y1, x):
        return k * (x - x1) + y1

    t_sample2 = np.linspace(t0, t1, int(100*(t1-t0)+1))
    q_spline2 = [linear_k(qd0, t0, q0, t) for t in t_sample2]
    t_sample2 = t_sample2[1: ]
    q_spline2 = q_spline2[1: ]

    # braking
    t0 = t_brake
    t1 = tf
    qd0 = qd1
    q0 = q_spline2[-1]
    q1 = qf
    qd1 = 0

    x = np.array([t0, t1])
    y = np.array([q0, q1])
    dydx = np.array([qd0, qd1])
    cubic_spline2 = CubicHermiteSpline(x, y, dydx)
    t_sample3 = np.linspace(t0, t1, int(100*(t1-t0)+1))
    q_spline3 = cubic_spline2(t_sample3)
    t_sample3 = t_sample3[1:]
    q_spline3 = q_spline3[1:]  

    t_total = np.concatenate((t_sample, t_sample2, t_sample3)) # assemble time array
    angle_bound = np.concatenate((q_spline, q_spline2, q_spline3)) # assemble angle matrix

    return t_total, angle_bound, t_accel, t_brake

start1 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end1 = np.array([pi, -pi/3, pi/3, -5*pi/6, -0.58*pi, -0.082*pi])

start2 = np.array([pi/2, -pi/2, pi/2, -pi/2, -pi/2, 0])
end2 = np.array([pi, -pi, 0, pi/4, -pi/2, pi])

start3 = np.array([0, 0, 0, 0, 0, 0])
end3 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])

start4 = np.array([pi, -pi/2, pi/2, -pi/2, -pi/2, 0])
end4 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start5 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end5 = np.array([2*pi/3, -pi/8, pi, -pi/2, 0, -pi/3])

#start = np.array([-pi, -pi, pi/2, -pi/2, -pi/2, 0])
#end = np.array([0, -0.749*pi, 0.69*pi, 0.444*pi, -0.8*pi, -pi])

joint = generate_traj_time(2, 201, start1, end1)
tolerance_band = np.zeros((6, 2, 201))
upper_bound = np.zeros(6)
time = joint[6]

for i in range(1): # generate tolerance band for all joints
    angle = joint[i].q
    velocity = joint[i].qd
    upper = create_tolerance_bands(angle, velocity, time, 0.65, "upper")
    #print(upper[1])
    lower = create_tolerance_bands(angle, velocity, time, 0.65, "lower")
    tolerance_band[i, 0, :] = upper[1]
    tolerance_band[i, 1, :] = lower[1]

np.savetxt("upper_tolerance.txt", upper[1])
np.savetxt("lower_tolerance.txt", lower[1])
np.savetxt("tolerance_band_time.txt", upper[0])

plt.plot(time, joint[0].q, label='Original joint trajectory', color='r')
plt.plot(upper[0], tolerance_band[0, 0, :], label='Tolerance band', linestyle='dashed', color='green')
plt.plot(upper[0], tolerance_band[0, 1, :], linestyle='dashed', color='green')
plt.legend(fontsize=9)
plt.xlabel('Trajectory time in s', fontsize=10, labelpad=4)
plt.ylabel('Joint angle in rad', fontsize=10, labelpad=4)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.show()
"""
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(time, joint[0].q, label='original joint1', color='red')
ax1.plot(upper[0], tolerance_band[0, :, 0], label='tolerance band', linestyle='dashed', color='green')

ax2.plot(time, joint[1].q, label='original joint2', color='red')
ax2.plot(upper[0], tolerance_band[0, :, 1], label='tolerance band', linestyle='dashed', color='green')
ax2.plot(upper[0], tolerance_band[1, :, 1], linestyle='dashed', color='green')
ax2.legend()
ax3.plot(time, joint[2].q, label='original joint3', color='red')
ax3.plot(upper[0], tolerance_band[0, :, 2], label='tolerance band', linestyle='dashed', color='green')
ax3.plot(upper[0], tolerance_band[1, :, 2], linestyle='dashed', color='green')
ax3.legend()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(time, joint[3].q, label='original joint1', color='red')
ax1.plot(upper[0], tolerance_band[0, :, 3], label='tolerance band', linestyle='dashed', color='green')
ax1.plot(upper[0], tolerance_band[1, :, 3], linestyle='dashed', color='green')
ax1.legend()
ax2.plot(time, joint[4].q, label='original joint2', color='red')
ax2.plot(upper[0], tolerance_band[0, :, 4], label='tolerance band', linestyle='dashed', color='green')
ax2.plot(upper[0], tolerance_band[1, :, 4], linestyle='dashed', color='green')
ax2.legend()
ax3.plot(time, joint[5].q, label='original joint3', color='red')
ax3.plot(upper[0], tolerance_band[0, :, 5], label='tolerance band', linestyle='dashed', color='green')
ax3.plot(upper[0], tolerance_band[1, :, 5], linestyle='dashed', color='green')
ax3.legend()

plt.show()
"""
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