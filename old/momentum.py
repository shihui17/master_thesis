'''
@author: Shihui Liu

The function total_center_of_mass() calculates the position of the center of mass of the whole robot in cartesian space.
Returns optimized trajectories and optimized energy consumption.
Results are additionally saved to result_q.txt, result_qd.txt and result_qdd.txt in root directory.
'''

from math import pi
import roboticstoolbox as rtb
import numpy as np
from traj import generate_traj_time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def total_center_of_mass(position_mass, mass_vec, i):
    m_total = np.linalg.norm(mass_vec, 1)
    m_X = 0
    m_Y = 0
    m_Z = 0
    for joint in range(6):
        m_X += mass_vec[joint] * position_mass[joint, 0, i]
        m_Y += mass_vec[joint] * position_mass[joint, 1, i]
        m_Z += mass_vec[joint] * position_mass[joint, 2, i]
    Xcm = m_X / m_total
    Ycm = m_Y / m_total
    Zcm = m_Z / m_total
    return np.array([Xcm, Ycm, Zcm])

def calculate_momentum(angle, velocity): # these 3 matrices should have the size of (201, 6)

    Yu = rtb.models.DH.Yu() # import Yu with all relevant geometric and mechanical data
    mass_vec = np.zeros(6)
    v_ee = np.zeros(201)

    for j in range(6): # read masses for each robot segment
        mass_vec[j] = Yu[j].m

    center_of_mass = Yu.r # center of mass position vector with respect to CS of segment
    trafo_mass_list = []

    for s in range(6): # generate homogenous transformation matrix from local CS of segment to center of mass, repeat 6 times for all segments
        trafo_mass = np.array([[1, 0, 0, center_of_mass[0, s]], [0, 1, 0, center_of_mass[1, s]], [0, 0, 1, center_of_mass[2, s]], [0, 0, 0, 1]])
        trafo_mass_list.append(trafo_mass)

    lin_ang_vel = np.zeros((6, 6, 201))
    lin_vel = np.zeros((6, 3, 201)) 
    ang_vel = np.zeros((6, 3, 201))
    lin_momentum = np.zeros((6, 3, 201))
    ang_momentum = np.zeros((6, 3, 201))
    lin_momentum_total = np.zeros((3, 201))
    mmt_abs = np.zeros((6, 201))
    position_mass = np.zeros((6, 3, 201))
    T_cm = np.zeros((201, 3))
    v_ee = np.zeros((6, 201))
    for i in range(201):
        pose_list = []
        jacobian_list = []
        poses = Yu.fkine_all(angle[i, :])
        v_ee[:, i] = Yu.jacob0(angle[i, :]) @ velocity[i, :]
        #print(poses)

        for pose in (poses):
            pose_list.append(np.array(pose))    
            
        for joint_num in range(6):
            joint_jacobian = np.zeros((6, joint_num+1))    
            vec_mass = pose_list[joint_num+1] @ trafo_mass_list[joint_num] @ np.array([0, 0, 0, 1])
            vec_mass = vec_mass[:3]
            position_mass[joint_num, :, i] = vec_mass

            for count in range(joint_num+1):

                translation = pose_list[joint_num+1] @ trafo_mass_list[joint_num] @ np.array([0, 0, 0, 1]) - pose_list[count] @ np.array([0, 0, 0, 1])
                translation = translation[:3]
                rotation = pose_list[count] @ np.array([0, 0, 1, 0])
                rotation = rotation[:3]
                joint_jacobian[:3, count] = np.cross(rotation, translation)
                joint_jacobian[3:6, count] = rotation

            jacobian_list.append(np.round(joint_jacobian, 6))

        T_cm[i, :] = total_center_of_mass(position_mass, mass_vec, i)

        for joint_num in range(6):
            #print(velocity[i, :joint_num+1])
            lin_ang = jacobian_list[joint_num] @ velocity[i, :joint_num+1]
            lin_vel[joint_num][:, i] = lin_ang[:3] # this is the matrix for the linear velocity of the center of mass of each link. Momentum can be calculated based on this matrix.
            lin_momentum[joint_num][:, i] = Yu[joint_num].m * lin_ang[:3]
            lin_momentum_total[:, i] += lin_momentum[joint_num][:, i]
            mmt_abs[joint_num, i] = np.linalg.norm(lin_momentum[joint_num][:, i], 2)
            ang_vel[joint_num][:, i] = lin_ang[3:6] # this is the matrix for the angular velocity of the center of mass of each link. Momentum can be calculated based on this matrix.
            lin_ang_vel[joint_num][:, i] = (jacobian_list[joint_num] @ velocity[i, :joint_num+1])
    
    abs_lin_vel = np.linalg.norm(lin_vel, 2, axis=1)
    max_lin_vel = np.amax(abs_lin_vel, axis=1)
    arg_max_lin = np.argmax(max_lin_vel)
    result = np.linalg.norm(lin_momentum_total, 2, axis=0)
    result2 = max_lin_vel * np.linalg.norm(mass_vec, 1)
    return np.max(result)
