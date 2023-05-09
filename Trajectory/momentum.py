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

def calculate_momentum(angle, velocity, accel): # these 3 matrices should have the size of (201, 6)

    Yu = rtb.models.DH.Yu() # import Yu with all relevant geometric and mechanical data
    #traj = generate_traj_time(2.5, 201)
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

    result = np.linalg.norm(lin_momentum_total, 2, axis=0)
    return np.max(result)
    #print(v_ee)
    #print(v_abs)
    #print(np.max(v_abs))
#xline = position_mass[max_joint, 0, :]
#yline = position_mass[max_joint, 1, :]
#zline = position_mass[max_joint, 2, :]
#print(lin_vel[max_joint, :, :])


"""
    xline = lin_max[:, 0]
    yline = lin_max[:, 1]
    zline = lin_max[:, 2]
    xline1 = position_mass[max_joint, 0, :]
    yline1 = position_mass[max_joint, 1, :]
    zline1 = position_mass[max_joint, 2, :]
    xline2 = T_cm[:, 0]
    yline2 = T_cm[:, 1]
    zline2 = T_cm[:, 2]

    ax = plt.axes(111, projection='3d')
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.plot3D(xline, yline, zline)
    #ax.plot3D(xline1, yline1, zline1, color='blue')
    ax.plot3D(xline2, yline2, zline2, color='red', linewidth=4)

    start = np.transpose(position_mass[max_joint, :, :])
    #print(start)
    #print(np.round(lin_max, 2))
    for i in range(len(lin_max)):
    #for i in range(5):
        #ax.quiver(start[i, 0], start[i, 1], start[i, 2], lin_max[i, 0], lin_max[i, 1], lin_max[i, 2], arrow_length_ratio=0.01, length=np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True')
        ax.quiver(T_cm[i, 0], T_cm[i, 1], T_cm[i, 2], lin_momentum_total[0, i], lin_momentum_total[1, i], lin_momentum_total[2, i], length=np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True', arrow_length_ratio=0.01, color='blue')
        #ax.quiver(T_cm[i, 0], T_cm[i, 1], T_cm[i, 2], lin_max[i, 0], lin_max[i, 1], lin_max[i, 2], arrow_length_ratio=0.01, length=np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True')
    #plt.show()

    #print(lin_momentum_total)
    result = np.linalg.norm(lin_momentum_total, 2, axis=0)
    print(result)
    #plt.plot(traj[6], result)
    plt.show()
    #for joint_num in range(6):
        #for joint 6:
    #translation = pose_list[5] @ np.array([0, 0, 0, 1])
    #print(translation)
    return np.max(result)
"""


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
