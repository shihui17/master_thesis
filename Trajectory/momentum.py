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


def joint_jacobian(jacobian, joint_num, qd, mass_matrix):

    joint_vel = jacobian[:, :joint_num+1] @ qd[:joint_num+1]
    joint_vel = joint_vel[:3]
    momentum_joint = mass_matrix[joint_num] * joint_vel

    return momentum_joint

angle = np.zeros((201, 6))
velocity = np.zeros((201, 6))
accel = np.zeros((201, 6))
mass = np.array([5.976, 9.572, 4.557, 2.588, 2.588, 1.025])
Yu = rtb.models.DH.Yu()
traj = generate_traj_time(2)
momentum = np.zeros(201)
for j in range(6):

    angle[:, j] = traj[j].q
    velocity[:, j] = traj[j].qd
    accel[:, j] = traj[j].qdd

#print(velocity)

for i in range(201):
    poses = Yu.fkine_all(angle[i, :])
    z_axis = []
    pos_vec = []
    translation = []
    jacobian = np.zeros((6, 6))

    for pose in poses:
        pose = np.array(pose)
        position = pose @ np.array([0, 0, 0, 1])
        position = position[:3]
        pos_vec.append(position)
        R_matrix = pose[:3, :3]
        rotation = R_matrix @ np.array([0, 0, 1])
        z_axis.append(rotation)
        translation.append(np.cross(rotation, position))

    for ii in range(1, 7):
        jacobian[ii-1, :] = np.hstack((translation[ii], z_axis[ii]))
    jacobian = np.transpose(np.round(jacobian, 3))

    qd = velocity[i, :]
    #print(velocity[i, :])
    velocity_cartesian = np.zeros(3)
    for joint_num in range(6):
        velocity_cartesian = velocity_cartesian + joint_jacobian(jacobian, joint_num, qd, mass)
        momentum_abs = np.linalg.norm(velocity_cartesian, 2)
    #momentum[i] = np.dot(velocity_cartesian, mass)
    print(momentum_abs)


"""        
poses = Yu.fkine_all([-pi/2, -pi/2, pi/2, -pi/3, -pi/3, 0]) # the pose of each robot joint as SE3
#print(p1)
#print(poses)
z_axis = []
pos_vec = []
translation = []
jacobian = np.zeros((6, 6))
for pose in poses:
    pose = np.array(pose)
    position = pose @ np.array([0, 0, 0, 1])
    position = position[:3]
    pos_vec.append(position)
    R_matrix = pose[:3, :3]
    rotation = R_matrix @ np.array([0, 0, 1])
    z_axis.append(rotation)
    translation.append(np.cross(rotation, position))
#print(z_axis)
print(translation)
for i in range(1, 7):
    jacobian[i-1, :] = np.hstack((translation[i], z_axis[i]))
jacobian = np.transpose(np.round(jacobian, 3))
#print(jacobian)
#print(jacobian[:, :2])
qd = np.array([1, 1, 1, 1, 1, 1])
#print(jacobian[:, :2] @ qd[:2])



velocity_cartesian = np.zeros((6, 6))

for joint_num in range(6):
    velocity_cartesian[:, joint_num] = joint_jacobian(jacobian, joint_num, qd)

print(velocity_cartesian)
"""