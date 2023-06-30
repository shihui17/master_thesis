"""
Plots the original total lin momentum and the optimized total lin momentum according to reflected mass optimization
"""

from math import pi
import roboticstoolbox as rtb
import numpy as np
from traj import generate_traj_time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

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

def calculate_momentum(traj, angle, velocity, accel):

    Yu = rtb.models.DH.Yu()
    #traj = generate_traj_time(2.5, 201)
    mass_vec = np.zeros(6)
    v_ee = np.zeros(201)

    for j in range(6):
        mass_vec[j] = Yu[j].m

    center_of_mass = Yu.r
    trafo_mass_list = []

    for s in range(6):
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
            lin_momentum[joint_num, :, i] = Yu[joint_num].m * lin_ang[:3] # linear momentum of each link
            lin_momentum_total[:, i] += lin_momentum[joint_num][:, i] # total linear momentum (sum of all links)
            mmt_abs[joint_num, i] = np.linalg.norm(lin_momentum[joint_num][:, i], 2)
            ang_vel[joint_num][:, i] = lin_ang[3:6] # this is the matrix for the angular velocity of the center of mass of each link. Momentum can be calculated based on this matrix.
            lin_ang_vel[joint_num][:, i] = (jacobian_list[joint_num] @ velocity[i, :joint_num+1])

    max_mmt = np.amax(mmt_abs, axis=1)
    #print(mmt_abs)
    #print(max_mmt)
    max_joint = np.argmax(max_mmt)
    #print(max_joint)
    max_joint_mmt = np.max(max_mmt)
    #print(max_mmt)
    lin_max = np.transpose(lin_momentum[max_joint, :, :])
    #print(lin_max)
    return lin_max, lin_momentum_total, position_mass, T_cm, max_joint

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
start1 = np.array([-pi, -pi/2, pi/2, -pi/2, -pi/2, 0])
end1 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start2 = np.array([pi/2, -pi/2, pi/2, -pi/2, -pi/2, 0])
end2 = np.array([pi, -pi, 0, pi/4, -pi/2, pi])

start3 = np.array([0, 0, 0, 0, 0, 0])
end3 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])

start4 = np.array([pi, -pi/2, pi/2, -pi/2, -pi/2, 0])
end4 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start5 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end5 = np.array([2*pi/3, -pi/8, pi, -pi/2, 0, -pi/3])

angle = np.zeros((6, 201))
velocity = np.zeros((6, 201))
accel = np.zeros((6, 201))
traj_opt = generate_traj_time(1.36, 201, start1, end1)

for j in range(6): # reads necessary data from the original trajectory
    angle[j, :] = traj_opt[j].q
    velocity[j, :] = traj_opt[j].qd
    accel[j, :] = traj_opt[j].qdd

angle = np.transpose(angle)
velocity = np.transpose(velocity)
accel = np.transpose(accel)
mass_vec = np.zeros(6)
angle_og = np.zeros((201, 6))
velocity_og = np.zeros((201, 6))
accel_og = np.zeros((201, 6))
Yu = rtb.models.DH.Yu()
traj = generate_traj_time(0.8, 201, start1, end1)

for j in range(6):
    angle_og[:, j] = traj[j].q
    velocity_og[:, j] = traj[j].qd
    accel_og[:, j] = traj[j].qdd
    mass_vec[j] = Yu[j].m

result = calculate_momentum(traj, angle, velocity, accel)
result_og = calculate_momentum(traj, angle_og, velocity_og, accel_og)

lin_max = result[0]
position_mass = result[2]
lin_momentum_total = result[1]
T_cm = result[3]
max_joint = result[4]

lin_max_og = result_og[0]
position_mass_og = result_og[2]
lin_momentum_total_og = result_og[1]
T_cm_og = result_og[3]
max_joint_og = result_og[4]


xline1 = T_cm_og[:, 0]
np.savetxt("x_orig.txt", xline1)
print(T_cm_og)
print(T_cm)
yline1 = T_cm_og[:, 1]
np.savetxt("y_orig.txt", yline1)
zline1 = T_cm_og[:, 2]
np.savetxt("z_orig.txt", zline1)
xline2 = T_cm[:, 0]
np.savetxt("x_opt.txt", xline2)
yline2 = T_cm[:, 1]
np.savetxt("y_opt.txt", yline2)
zline2 = T_cm[:, 2]
np.savetxt("z_opt.txt", zline2)


ax = plt.axes(111, projection='3d')
#ax.set_xlim([-8, 8])
#ax.set_ylim([-8, 8])
#ax.set_zlim([-1, 1])
ax.set_xlabel('x coordinate in m')
ax.set_ylabel('Y coordinate in m')
ax.set_zlabel('Z coordinate in m')
#ax.plot3D(xline, yline, zline)
#ax.plot3D(xline1, yline1, zline1, color='blue')
ax.plot3D(xline1, yline1, zline1, color='red', linewidth=1, label='Original trajectory of center of mass')
ax.plot3D(xline2, yline2, zline2, color='green', linewidth=1, label='Optimized rajectory of center of mass')
ax.legend()
plt.savefig('C:\Codes\master_thesis\Trajectory\Figures\Momentum/mass_traj.png')
plt.show()

start = np.transpose(position_mass[max_joint, :, :])
#print(start)
#print(np.round(lin_max, 2))
ax2 = plt.axes(111, projection='3d')
ax2.set_xlabel('x coordinate in m')
ax2.set_ylabel('Y coordinate in m')
ax2.set_zlabel('Z coordinate in m')
ax2.plot3D(xline1, yline1, zline1, color='red', linewidth=1, label='Original trajectory of center of mass')
ax2.plot3D(xline2, yline2, zline2, color='green', linewidth=1, label='Optimized rajectory of center of mass')

for i in range(201):
#for i in range(5):
    if i == 0:
        ax2.quiver(T_cm_og[i, 0], T_cm_og[i, 1], T_cm_og[i, 2], lin_momentum_total_og[0, i], lin_momentum_total_og[1, i], lin_momentum_total_og[2, i], color='red', length=0.008*np.linalg.norm((start[i, :]-lin_momentum_total_og[:, i]), 2), normalize='True', arrow_length_ratio=0.02, label='Momentum vector of robot on its center of mass')
        ax2.quiver(T_cm[i, 0], T_cm[i, 1], T_cm[i, 2], lin_momentum_total[0, i], lin_momentum_total[1, i], lin_momentum_total[2, i], color='green', length=0.008*np.linalg.norm((start[i, :]-lin_momentum_total[:, i]), 2), normalize='True', arrow_length_ratio=0.02, label='Momentum vector of robot on its center of mass')
    elif i % 4 == 0:
    #ax.quiver(start[i, 0], start[i, 1], start[i, 2], lin_max[i, 0], lin_max[i, 1], lin_max[i, 2], arrow_length_ratio=0.01, length=np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True')
        ax2.quiver(T_cm_og[i, 0], T_cm_og[i, 1], T_cm_og[i, 2], lin_momentum_total_og[0, i], lin_momentum_total_og[1, i], lin_momentum_total_og[2, i], color='red', length=0.008*np.linalg.norm((start[i, :]-lin_momentum_total_og[:, i]), 2), normalize='True', arrow_length_ratio=0.02)#length=0.01*np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True', arrow_length_ratio=0.05, color='blue')    
        ax2.quiver(T_cm[i, 0], T_cm[i, 1], T_cm[i, 2], lin_momentum_total[0, i], lin_momentum_total[1, i], lin_momentum_total[2, i], color='green', length=0.008*np.linalg.norm((start[i, :]-lin_momentum_total[:, i]), 2), normalize='True', arrow_length_ratio=0.02)#length=0.01*np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True', arrow_length_ratio=0.05, color='blue')
#plt.show()
ax2.legend()
plt.savefig('C:\Codes\master_thesis\Trajectory\Figures\Momentum/momentum_traj.png')
plt.show()

result = np.linalg.norm(lin_momentum_total, 2, axis=0)
max = np.max(result)
argmax = np.argmax(result)
t_max = traj[6][argmax]

result_og = np.linalg.norm(lin_momentum_total_og, 2, axis=0)
np.savetxt("mmt_og_curve.txt", result_og)
np.savetxt("mmt_opt_curve.txt", result)
np.savetxt("mmt_time.txt", traj[6])
max_og = np.max(result_og)
argmax_og = np.argmax(result_og)
t_max_og = traj[6][argmax_og]

plt.plot(traj_opt[6], result, color='green', label='Optimized robot momentum')
plt.plot(traj[6], result_og, color='red', label='Original robot momtenum')
plt.plot(t_max, max, marker='o', markeredgecolor='green', markerfacecolor='green', label='Optimized maximal momentum')
plt.plot(t_max_og, max_og, marker='o', markeredgecolor='red', markerfacecolor='red', label='Original maximal momentum')
plt.xlabel('Trajectory time in s')
plt.ylabel('Robot total momentum in kg*m/s')
#plt.title('Change of total momentum over time', fontsize=20, pad=20)
plt.legend()
plt.savefig('C:\Codes\master_thesis\Trajectory\Figures\Momentum/momentum_total.png')
plt.show()
