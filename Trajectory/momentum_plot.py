'''
@author: Shihui Liu

The function total_center_of_mass() calculates the position of the center of mass of the whole robot in cartesian space.
The function calculate_momentum() calculates a series of momentum-related results given a robot trajectory in joint space.
The following data sets are plotted:
    - the trajectories of robot center of mass in cartesian space, pre- and post-optimization
    - the momentum vectors with respect to robot center of mass in cartesian space
    - the momentum curve over trajectory time, with respect to robot center of mass
'''
from math import pi
import roboticstoolbox as rtb
import numpy as np
from traj import generate_traj_time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

def total_center_of_mass(position_mass, mass_vec, time_step):
    """
    Calculates robot center of mass in cartesian space with respect to RF0, given the center of mass trajectories of all six robot joints

    :param position_mass: the position of center of mass of a given joint, defined as np.array([joint_num, 3, time_step]), the x, y and z coordinate are written in columns
    :param mass_vec: 1x6 numpy array. Stores the masses of the 6 joints
    :param time_step: time step, number of time increments over total trajectory time

    :return:
    :np.array([Xcm, Ycm, Zcm]): the position of robot center of mass in cartesian space with respect to RF0
    """    
    m_total = np.linalg.norm(mass_vec, 1) # total mass of the robot
    m_X = 0 # cumulative sum of x-coordinate of center of mass
    m_Y = 0 # y-coordinate
    m_Z = 0 # z-coordinate
    for joint in range(6):
        m_X += mass_vec[joint] * position_mass[joint, 0, time_step] # mass times x-coordinate of mass
        m_Y += mass_vec[joint] * position_mass[joint, 1, time_step] # mass times y-coordinate of mass
        m_Z += mass_vec[joint] * position_mass[joint, 2, time_step] # mass times z-coordinate of mass
    Xcm = m_X / m_total # resulting x-coordinate of mass
    Ycm = m_Y / m_total # resulting y-coordinate of mass
    Zcm = m_Z / m_total # resulting z-coordinate of mass
    return np.array([Xcm, Ycm, Zcm])

def calculate_momentum(traj):

    """
    does some momentum related calculations given a robot trajectory class, can return following calculation results as required:
        - max_momentum: the maximum momentum that occurs during trajectory time, with respect to robot center of mass and RF0
        - lin_vel: the linear velocity of each robot segment, with respect to center of mass of each segment, 3x1 numpy array
        - ang_vel: the angular velocity of each robot segment, with respect to center of mass of each segment, 3x1 numpy array
        - lin_momentum: the linear momentum of each joint over entire trajectory, 6 x 3 x time_step numpy array
        - lin_momentum_total: the linear momentum of the robot, with respect to robot center of mass, 3 x time_step numpy array
    The following data sets are plotted:
        - the trajectories of robot center of mass in cartesian space, pre- and post-optimization
        - the momentum vectors with respect to robot center of mass in cartesian space
        - the momentum curve over trajectory time, with respect to robot center of mass
    :param traj: robot trajectory in joint space, see traj.py
    """    
    angle = np.zeros((201, 6))
    velocity = np.zeros((201, 6))
    accel = np.zeros((201, 6))
    Yu = rtb.models.DH.Yu()
    mass_vec = np.zeros(6)
    v_ee = np.zeros(201)

    for j in range(6): # fill the matrices initialized above with corresponding trajectory data and masses of all robot segments
        angle[:, j] = traj[j].q 
        velocity[:, j] = traj[j].qd
        accel[:, j] = traj[j].qdd
        mass_vec[j] = Yu[j].m

    center_of_mass = Yu.r # transformation between the RF-basis of robot segment and the center of mass of the segment
    trafo_mass_list = [] # initialize a list that contains transformation matrices for all 6 segments

    for s in range(6):
        trafo_mass = np.array([[1, 0, 0, center_of_mass[0, s]], [0, 1, 0, center_of_mass[1, s]], [0, 0, 1, center_of_mass[2, s]], [0, 0, 0, 1]]) # assemble a homogeneous transformation matrix
        trafo_mass_list.append(trafo_mass) # write to the list to be used later

    lin_ang_vel = np.zeros((6, 6, 201)) # combined velocity of robot segments, directly calculated through their Jacobian matrices
    lin_vel = np.zeros((6, 3, 201)) # linear velocity of robot segments, upper half of lin_ang_vel
    ang_vel = np.zeros((6, 3, 201)) # angular velocity of robot segments, lower half of lin_ang_vel
    lin_momentum = np.zeros((6, 3, 201)) # linear momentum of robot segments 
    ang_momentum = np.zeros((6, 3, 201)) # angular momentum of robot segments
    lin_momentum_total = np.zeros((3, 201)) # total linear momentum of robot, sum of the linear momentum of each robot segment
    mmt_abs = np.zeros((6, 201)) # absolute value of linear momentum, 2-Norm of lin_momentum 
    position_mass = np.zeros((6, 3, 201)) # position of center of mass of each robot segment in cartesian space, with respect to RF0
    T_cm = np.zeros((201, 3)) # robot center of mass
    v_ee = np.zeros((6, 201)) # end effector velocity

    for i in range(201): # iterate over all time increments
        pose_list = [] # initialize list of direct kinematic matrices of robot segments
        jacobian_list = [] # initialize list of Jacobian matrices of robot segments
        poses = Yu.fkine_all(angle[i, :]) # direct kinematic of each robot segment with respect to RF0
        v_ee[:, i] = Yu.jacob0(angle[i, :]) @ velocity[i, :] # calculate end effector velocity

        for pose in (poses): # convert SE3 instance to numpy array
            pose_list.append(np.array(pose))    
            
        for joint_num in range(6): # calculate Jacobian matrix for each robot segment
            joint_jacobian = np.zeros((6, joint_num+1))    
            vec_mass = pose_list[joint_num+1] @ trafo_mass_list[joint_num] @ np.array([0, 0, 0, 1]) # position vector from segment RF to center of mass of the segment
            vec_mass = vec_mass[:3] # delete the "1" in last row
            position_mass[joint_num, :, i] = vec_mass # write result to matrix to be used later

            for count in range(joint_num+1): # assemble the Jacobian matrix for each robot segment, starting from segment 1
                translation = pose_list[joint_num+1] @ trafo_mass_list[joint_num] @ np.array([0, 0, 0, 1]) - pose_list[count] @ np.array([0, 0, 0, 1]) # position vector from RF_count to center of mass of target segment
                translation = translation[:3] # deleting the last element "1"
                rotation = pose_list[count] @ np.array([0, 0, 1, 0]) # rotation axis of the base segment
                rotation = rotation[:3] # deleting "1"
                joint_jacobian[:3, count] = np.cross(rotation, translation) # assemble the translational part of Jacobian matrix
                joint_jacobian[3:6, count] = rotation # assemble the rotational part of Jacobian matrix

            jacobian_list.append(np.round(joint_jacobian, 6)) # store the calculated Jacobian to a list

        T_cm[i, :] = total_center_of_mass(position_mass, mass_vec, i) # store center of mass position of the robot

        for joint_num in range(6):
            lin_ang = jacobian_list[joint_num] @ velocity[i, :joint_num+1] # v = J * qd
            lin_vel[joint_num][:, i] = lin_ang[:3] # this is the matrix for the linear velocity of the center of mass of each link. Momentum can be calculated based on this matrix.
            lin_momentum[joint_num][:, i] = Yu[joint_num].m * lin_ang[:3] # momentum = mass * velocity
            lin_momentum_total[:, i] += lin_momentum[joint_num][:, i] # hypothesis: total momentum = sum of momentum of each segment
            mmt_abs[joint_num, i] = np.linalg.norm(lin_momentum[joint_num][:, i], 2) # scalar momentum of robot, 2-Norm of linear momentum
            ang_vel[joint_num][:, i] = lin_ang[3:6] # this is the matrix for the angular velocity of the center of mass of each link. Momentum can be calculated based on this matrix.
            lin_ang_vel[joint_num][:, i] = (jacobian_list[joint_num] @ velocity[i, :joint_num+1])

    max_mmt = np.amax(mmt_abs, axis=1)
    max_joint = np.argmax(max_mmt)
    #max_joint_mmt = np.max(max_mmt)
    #print(max_mmt)
    lin_max = np.transpose(lin_momentum[max_joint, :, :])
    #print(lin_max)
    
    """
    v_ee = v_ee[:3, :]
    v_ee = np.transpose(v_ee)
    v_abs = np.linalg.norm(v_ee, 2, axis=1)
    print(v_ee)
    print(v_abs)
    print(np.max(v_abs))
    """
#xline = position_mass[max_joint, 0, :]
#yline = position_mass[max_joint, 1, :]
#zline = position_mass[max_joint, 2, :]
#print(lin_vel[max_joint, :, :])



    xline = lin_max[:, 0]
    yline = lin_max[:, 1]
    zline = lin_max[:, 2]
    xline1 = position_mass[0, 0, :]
    yline1 = position_mass[0, 1, :]
    zline1 = position_mass[0, 2, :]
    xline2 = T_cm[:, 0]
    yline2 = T_cm[:, 1]
    zline2 = T_cm[:, 2]


    ax = plt.axes(111, projection='3d')
    #ax.set_xlim([-8, 8])
    #ax.set_ylim([-8, 8])
    #ax.set_zlim([-1, 1])
    ax.set_xlabel('x coordinate in m')
    ax.set_ylabel('Y coordinate in m')
    ax.set_zlabel('Z coordinate in m')
    #ax.plot3D(xline, yline, zline)
    #ax.plot3D(xline1, yline1, zline1, color='blue')
    ax.plot3D(xline2, yline2, zline2, color='red', linewidth=1, label='Trajectory of center of mass')
    plt.savefig('C:\Codes\master_thesis\Trajectory\Figures\Momentum/mass_traj.png')
    start = np.transpose(position_mass[max_joint, :, :])
    #print(start)
    #print(np.round(lin_max, 2))
    for i in range(len(lin_max)):
    #for i in range(5):
        if i == 0:
            ax.quiver(T_cm[i, 0], T_cm[i, 1], T_cm[i, 2], lin_momentum_total[0, i], lin_momentum_total[1, i], lin_momentum_total[2, i], color='green', length=0.1*np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True', arrow_length_ratio=0.05, label='Momentum vector of robot on its center of mass')
        elif i % 4 == 0:
        #ax.quiver(start[i, 0], start[i, 1], start[i, 2], lin_max[i, 0], lin_max[i, 1], lin_max[i, 2], arrow_length_ratio=0.01, length=np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True')
            ax.quiver(T_cm[i, 0], T_cm[i, 1], T_cm[i, 2], lin_momentum_total[0, i], lin_momentum_total[1, i], lin_momentum_total[2, i], color='green', length=0.1*np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True', arrow_length_ratio=0.05)#length=0.01*np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True', arrow_length_ratio=0.05, color='blue')
        #ax.quiver(T_cm[i, 0], T_cm[i, 1], T_cm[i, 2], lin_max[i, 0], lin_max[i, 1], lin_max[i, 2], arrow_length_ratio=0.01, length=np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True')
    #plt.show()
    ax.legend()
    plt.savefig('C:\Codes\master_thesis\Trajectory\Figures\Momentum/momentum_traj.png')
    #print(lin_momentum_total)
    result = np.linalg.norm(lin_momentum_total, 2, axis=0)
    #print(result)
    max = np.max(result)
    argmax = np.argmax(result)
    t_max = traj[6][argmax]
    #plt.plot(traj[6], result)
    plt.show()
    #for joint_num in range(6):
        #for joint 6:
    #translation = pose_list[5] @ np.array([0, 0, 0, 1])
    #print(translation)
    plt.plot(traj[6], result)
    plt.plot(t_max, max, marker='o', markeredgecolor='blue')
    plt.xlabel('Trajectory time in s')
    plt.ylabel('Robot total momentum in kg*m/s')
    plt.title('Change of total momentum over time', fontsize=20)
    plt.savefig('C:\Codes\master_thesis\Trajectory\Figures\Momentum/momentum_total.png')
    plt.show()

    return 
start1 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end1 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start2 = np.array([pi/2, -pi/2, pi/2, -pi/2, -pi/2, 0])
end2 = np.array([pi, -pi, 0, pi/4, -pi/2, pi])

start3 = np.array([0, 0, 0, 0, 0, 0])
end3 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])

start4 = np.array([pi, -pi/2, pi/2, -pi/2, -pi/2, 0])
end4 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start5 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end5 = np.array([2*pi/3, -pi/8, pi, -pi/2, 0, -pi/3])

#end = np.array([-pi, -pi/2, pi/2, -pi/2, -pi/2, 0])
traj = generate_traj_time(2, 201, start1, end1)
calculate_momentum(traj)

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
""" 
    Evaluate each link
    lin_v = np.zeros((6, 201))
    for ii in range(6):
        lin_v[ii, :] = np.linalg.norm(lin_momentum[ii, :, :], axis=0)
    lin_r = np.linalg.norm(lin_momentum_total, axis=0)
    print(np.amax(lin_v, axis=1))
    print(lin_momentum_total)
    print(lin_r)
"""
    
    #print(np.linalg.norm(lin_momentum, axis=0))
#print(lin_vel) # this is the matrix for the linear velocity of the center of mass of each link. Momentum can be calculated based on this matrix.
#print(ang_vel) 
#lin_momentum_total = np.transpose(lin_momentum_total)
#print(position_mass)
#print(T_cm)


#plt.show()
#print(lin_vel[4, :, :])
#print(np.transpose(lin_momentum[3, :, :]))
#print(lin_momentum_total)
#print(mmt_abs)