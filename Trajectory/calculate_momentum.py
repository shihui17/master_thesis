import roboticstoolbox as rtb
import numpy as np

def calculate_momentum(q, qd, point, joint_num): 
    """
    Calculate the instantaneous linear velocity and reflected mass of the contact point at a given time t, their product is the linear momentum of the point

    :param q: instantaneous angle of each joint at a given time t, size: 6x1
    :param qd: instantaneous velocity of each joint at a given time t, size: 6x1
    :param point: the contact point to be calculated
    :return vel_u: the unit vector of the linear velocity at contact point
    :return m_refl: reflected mass at contact point
    :return lin_vel: instantaneous linear velocity of contact point at time t
    :return rf_p: position vector of the contact point for visualisation, not used anymore
    """
    Yu = rtb.models.DH.Yu() # import Yu with all relevant geometric and mechanical data

    trafo_p = np.eye(4) # initialise homogeneous transformation matrix
    trafo_p[:, -1] = point # last column of the matrix is the contact point
    joint_jacobian = np.zeros((6, 6)) # initialise modified Jacobian
    pose_list = [] # a list of forward kinematics of each robot segment
    poses = Yu.fkine_all(q) # calculate forward kinematics of each robot segment with respect to robot basis

    for pose in (poses): # convert forward kinematics (SE3) into numpy arrays
        pose_list.append(np.array(pose))  

    rf_p =  pose_list[joint_num+1] @ trafo_p @ np.array([0, 0, 0, 1]) # Position vector of contact point expressed in RF0

    for count in range(joint_num+1): # calculation of modified Jacobian, iterate through each robot segment, from robot basis to the robot segment where the contact point is located
        translation = rf_p - pose_list[count] @ np.array([0, 0, 0, 1]) # Position vector from RF of robot segment to the contact point
        translation = translation[:3] # Remove the last "1" element
        rotation = pose_list[count] @ np.array([0, 0, 1, 0]) # Rotation axis of the robot segment
        rotation = rotation[:3] # remove the last "0" element
        joint_jacobian[:3, count] = np.cross(rotation, translation) # translation part of the modified Jacobian
        joint_jacobian[3:6, count] = rotation # rotation part of the modified Jacobian

    lin_ang = joint_jacobian @ qd # linear and angular velocity of the contact point
    lin_vel = lin_ang[:3] # linear velocity of the contact point
    velocity_abs = np.linalg.norm(lin_vel, 2) # absolute value of linear velocity
    if velocity_abs <= 1e-3: # if the velocity is sufficiently small
        vel_u = np.zeros(3) # assume the contact point is stationary
    else:
        vel_u = np.divide(lin_vel, (np.linalg.norm(lin_vel, 2))) # calculate unit vector

    M_q = Yu.inertia(q) # inertia matrix of the robot
    Aq_inv = joint_jacobian @ np.linalg.inv(M_q) @ np.transpose(joint_jacobian) # inverse of the effective inertia matrix of the contact point

    if all(element == 0 for element in vel_u): # if point is stationary
        m_refl = 0 # set reflected mass to zero to avoid division over zero
    else:
        m_refl = 1/(np.transpose(vel_u) @ Aq_inv[:3, :3] @ vel_u) # calculate reflected mass of the contact point

    return vel_u, m_refl, lin_vel, rf_p