from math import pi
import roboticstoolbox as rtb
import numpy as np
from traj import generate_traj_time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

Yu = rtb.models.DH.Yu()

traj = generate_traj_time(2.5, 201)
angle = np.zeros((6, 201))
velocity = np.zeros((6, 201))
accel = np.zeros((6, 201))
mass_vec = np.zeros(6)
points = [np.array([0, -0.13687, 0, 1]), np.array([-0.450, 0, 0.13687, 1]), np.array([0, 0, -0.1957, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0, 1])] # points of interest


for j in range(6):

    angle[j, :] = traj[j].q
    velocity[j, :] = traj[j].qd
    accel[j, :] = traj[j].qdd
    mass_vec[j] = Yu[j].m

mass = np.linalg.norm(mass_vec, 1) # total mass of robot
# define a point on the robot
r1 = np.array([-0.450, 0, 0.13687, 1]) # Position vector of p1 relative to RF2 (segment 2)


def unit_vector(q, point, joint_num, t, qd): # angle and velocity matrix should have the size of (201, 6), point = np.array([x, y, z, 1]), t: traj time

    Yu = rtb.models.DH.Yu() # import Yu with all relevant geometric and mechanical data
    #traj = generate_traj_time(2.5, 201)

    trafo_p = np.eye(4)
    trafo_p[:, -1] = point
    #lin_vel = np.zeros((6, 3, 201)) 
    vel_um = np.zeros((6, 3, 201))
    joint_jacobian = np.zeros((6, joint_num+1))   
    #lin_momentum = np.zeros((6, 3, 201))
    #ang_momentum = np.zeros((6, 3, 201))
    #lin_momentum_total = np.zeros((3, 201))
    #mmt_abs = np.zeros((6, 201))
    #position_mass = np.zeros((6, 3, 201))
    #T_cm = np.zeros((201, 3))
    #v_ee = np.zeros((6, 201))
    pose_list = []
    poses = Yu.fkine_all(q)


    for pose in (poses):
        pose_list.append(np.array(pose))  

    #print(trafo_p)
    rf_p = pose_list[joint_num+1] @ trafo_p @ np.array([0, 0, 0, 1]) # Position vector of p in RF0

    for count in range(joint_num+1):
        translation = rf_p - pose_list[count] @ np.array([0, 0, 0, 1])
        translation = translation[:3]
        rotation = pose_list[count] @ np.array([0, 0, 1, 0])
        rotation = rotation[:3]
        joint_jacobian[:3, count] = np.cross(rotation ,translation)
        joint_jacobian[3:6, count] = rotation

    lin_ang = joint_jacobian @ qd[:joint_num+1] # velocity[t, :joint_num+1]
    lin_vel = lin_ang[:3]
    #print(lin_vel)
    length = np.linalg.norm(lin_vel, 2)
    if length <= 1e-3:
        vel_u = np.zeros(3)
    else:
        vel_u = np.divide(lin_vel, (np.linalg.norm(lin_vel, 2)))

    fkine_ee = pose_list[-1]
    r6_0 = fkine_ee[:, -1]
    #print(rf_p)
    #print(r6_0)
    r6p_0 = rf_p - r6_0 # Position vector from RF6 to p in RF0
    #print(r6p_0)
    T_0e = Yu.fkine(q)
    #print(T_0e)
    T_e0 = np.linalg.inv(T_0e)
    #print(T_e0)
    r6p_0 = T_e0 @ r6p_0
    #print(r6p_0)
    trafo_6p = np.eye(4)
    trafo_6p[:3, -1] = r6p_0[:3]
    #print(trafo_6p)

    Yu.tool = trafo_6p
    jacobian_p = Yu.jacob0(q)
    M_q = Yu.inertia(q)
    Aq_inv = jacobian_p @ np.linalg.inv(M_q) @ np.transpose(jacobian_p)
    #print(jacobian_p)
    #print(f'vel = {vel_u}\n')
    if all(element == 0 for element in vel_u):
        m_refl = 0
        #print('oh no')
    else:
        #print(vel_u)
        m_refl = 1/(np.transpose(vel_u) @ Aq_inv[:3, :3] @ vel_u)
        #print('oh yes')
    #print(f'reflected mass at point: {m_refl} kg')
    #print(f'mass of the whole robot: {mass} kg')

    return vel_u, m_refl, lin_vel, rf_p

m_refl = np.zeros((6, 201))
lin_vel = np.zeros((6, 3, 201))
rf_p_storage = np.zeros((6, 3, 201))


for t in range(201):
    q = angle[:, t]
    qd = velocity[:, t]

    for count, r in enumerate(points):
        #print(f'r = {r}\n')
        result = unit_vector(q, r, count, t, qd)
        lin_vel[count, :, t] = result[2]
        m_refl[count, t] = result[1]
        rf_p_storage[count, :, t] = result[3][:3]

n = 4
print(m_refl[n, :])
#print(lin_vel[3, :, :])
refl_max = np.argmax(m_refl[n, :])
print(refl_max)
print(m_refl[n, refl_max])
ax = plt.axes(111, projection='3d')

lin_vel_plot = lin_vel[n, :, :]
start = np.transpose(rf_p_storage[n, :, :])

xline2 = start[:, 0]
yline2 = start[:, 1]
zline2 = start[:, 2]


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
#plt.show()

#print(start)
#print(np.round(lin_max, 2))
for i in range(201):
#for i in range(5):
    if i == 0:
        ax.quiver(start[i, 0], start[i, 1], start[i, 2], lin_vel_plot[0, i], lin_vel_plot[1, i], lin_vel_plot[2, i], color='green', arrow_length_ratio=0, label='Momentum vector of robot on its center of mass')
    elif i % 4 == 0:
    #ax.quiver(start[i, 0], start[i, 1], start[i, 2], lin_max[i, 0], lin_max[i, 1], lin_max[i, 2], arrow_length_ratio=0.01, length=np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True')
        ax.quiver(start[i, 0], start[i, 1], start[i, 2], lin_vel_plot[0, i], lin_vel_plot[1, i], lin_vel_plot[2, i], color='green', arrow_length_ratio=0)#length=0.01*np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True', arrow_length_ratio=0.05, color='blue')
    #ax.quiver(T_cm[i, 0], T_cm[i, 1], T_cm[i, 2], lin_max[i, 0], lin_max[i, 1], lin_max[i, 2], arrow_length_ratio=0.01, length=np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True')
#plt.show()
#ax.legend()
plt.show()