import numpy as np
import roboticstoolbox as rtb
from traj import generate_traj_time
import matplotlib.pyplot as plt


def joint_jacobian(jacobian, joint_num, qd, mass_matrix):

    joint_vel = jacobian[:, :joint_num+1] @ qd[:joint_num+1]
    joint_vel = joint_vel[:3]
    momentum_joint = mass_matrix[joint_num] * joint_vel

    return momentum_joint

angle = np.zeros((401, 6))
velocity = np.zeros((401, 6))
accel = np.zeros((401, 6))
mass = np.array([5.976, 9.572, 4.557, 2.588, 2.588, 1.025])
Yu = rtb.models.DH.Yu()
traj = generate_traj_time(2, 401)
#plot_trajectory(traj)
time = traj[6]
momentum = np.zeros(401)
momentum_abs = []
momentum_matrix = np.zeros((6, 401))
for j in range(6):

    angle[:, j] = traj[j].q
    velocity[:, j] = traj[j].qd
    accel[:, j] = traj[j].qdd

#print(velocity)
for i in range(50):
    #poses = Yu.fkine_all(np.array([0, 0, 0, 0, 0, 0]))
    poses = Yu.fkine_all(angle[i, :])
    #print(poses)
    z_axis = []
    pos_vec = []
    translation = []
    jacobian = np.zeros((6, 6))

    for pose in poses:
        pose = np.array(pose)
        #print(pose)
        position = pose @ np.array([0, 0, 0, 1])
        position = position[:3]
        pos_vec.append(position)
        R_matrix = pose[:3, :3]
        rotation = R_matrix @ np.array([0, 0, 1])
        z_axis.append(rotation)
        translation.append(np.cross(rotation, position))

    print(translation)
    #print(z_axis)

    for ii in range(1, 7):
        jacobian[ii-1, :] = np.hstack((translation[ii], z_axis[ii]))
    jacobian = np.transpose(np.round(jacobian, 3))
    print(jacobian)

    qd = velocity[i, :]
    #print(velocity[i, :])
    momentum_cartesian = np.zeros(3)
    for joint_num in range(6):
        joint_jacob = joint_jacobian(jacobian, joint_num, qd, mass)
        joint_jacob_abs = np.linalg.norm(joint_jacob, 2)
        momentum_matrix[joint_num, i] = joint_jacob_abs
        momentum_cartesian = momentum_cartesian + joint_jacob
    momentum_abs.append(np.linalg.norm(momentum_cartesian, 2))
    #momentum[i] = np.dot(velocity_cartesian, mass)
#print(momentum_abs)
#print(len(momentum_abs))
#print(time)
#print(max(momentum_abs))
"""
plt.plot(time, momentum_abs)
plt.title('Total Robot Momentum', fontsize=16)
plt.xlabel(f'Trajectory Time in s', fontsize=10, labelpad=10)
plt.ylabel('Total robot momentum in kg*m/s', fontsize=10, labelpad=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
"""
"""
fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained')
ax1.plot(time, momentum_abs, label='total')
ax2.plot(time, momentum_matrix[0, :], label='joint 1')
ax2.plot(time, momentum_matrix[1, :], label='joint 2')
ax2.plot(time, momentum_matrix[2, :], label='joint 3')
ax2.plot(time, momentum_matrix[3, :], label='joint 4')
ax2.plot(time, momentum_matrix[4, :], label='joint 5')
ax2.plot(time, momentum_matrix[5, :], label='joint 6')
plt.legend()
plt.show()
"""
#print(momentum_matrix[1, :])
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