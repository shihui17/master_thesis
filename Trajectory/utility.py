'''
@author: Shihui Liu

Contains some utility functions that are used in force/momentum optimization
'''

import roboticstoolbox as rtb
import numpy as np

def get_contact_points():
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.44995, 0, 0.12128, 1]), np.array([0.01559, 0, 0.12128, 1]), np.array([0.05, 0, 0.07938, 1]), 
              np.array([0, -0.05, 0.07938, 1]), np.array([-0.05, 0, 0.07938, 1]), np.array([0, 0, 0, 1])] # all contact points are defined here
    point_tuple_def = [('Point Coordinate', np.ndarray), ('Segment','i2')] 
    point_tuple = np.zeros((7), dtype = point_tuple_def)
    for i, p in enumerate(points):
        point_tuple[i][0] = p

    point_tuple[0][1] = 0 # the robot segment index where the contact point is located
    point_tuple[1][1] = 1
    point_tuple[2][1] = 1
    point_tuple[3][1] = 2
    point_tuple[4][1] = 3
    point_tuple[5][1] = 4
    point_tuple[6][1] = 5

    return point_tuple

def get_original_jointdata(trajectory_data):
    """
    Generate original trajectory data using tools.trapezoidal method from roboticstoolbox, return the trajectory data as numpy arrays

    :param trajectory_data: trajectory data (as a python class) calculated by roboticstoolbox
    :return angle: angle trajectory of all joints
    :return velocity: velocity trajectory of all joints
    :return accel: acceleration trajectory of all joints
    :return mass: total mass of the robot
    :return q0: initial joint configuration
    :return qf: final joint configuration
    """
    Yu = rtb.models.DH.Yu()
    mass_vec = np.zeros(6)
    angle = np.zeros((6, 201))
    velocity = np.zeros((6, 201))
    accel = np.zeros((6, 201))
    q0 = np.zeros(6)
    qf = np.zeros(6)

    for j in range(6):
        angle[j, :] = trajectory_data[j].q
        velocity[j, :] = trajectory_data[j].qd
        accel[j, :] = trajectory_data[j].qdd
        mass_vec[j] = Yu[j].m
        q0[j] = angle[j, 0]
        qf[j] = angle[j, -1]
               
    mass = np.linalg.norm(mass_vec, 1) # total mass of robot
    return angle, velocity, accel, mass, q0, qf

def gen_traj(vel_max, t_accel_rand, q0, qf, tf, time, V=None):
    """
    Generate a trapezoidal trajectory in joint space based on parameters generated by HKA
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
    else: # if specified, check its feasibility
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