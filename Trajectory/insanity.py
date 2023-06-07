from math import pi
import roboticstoolbox as rtb
import numpy as np
from traj import generate_traj_time
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


Yu = rtb.models.DH.Yu()
case = 0
if case == 0: # zero-pose, robot rotates around axis 1 for pi/2
    start = np.array([pi/2, 0, 0, 0, 0, 0])
    end = np.array([0, 0, 0, -0, -0, 0])    
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.44995, 0, 0.12128, 1]), np.array([0.01559, 0, 0.12128, 1]), np.array([0.05, 0, 0.07938, 1]), 
              np.array([0, -0.05, 0.07938, 1]), np.array([-0.05, 0, 0.07938, 1]), np.array([0, 0, 0, 1])]
elif case == 1: # reverse of case 0
    start = np.array([-0, 0, 0, 0, 0, 0])
    end = np.array([pi/2, 0, 0, -0, -0, 0])
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.50519, 0, 0.12128, 1]), np.array([0, 0, 0.13687, 1]), np.array([0, 0, 0.10946, 1]), 
            np.array([0, 0.05, 0.07938, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0, 1])]
elif case == 2: 
    start = np.array([-0, -pi/2, 0, 0, 0, 0])
    end = np.array([0, 0, 0, -0, -0, 0])
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.50519, 0, 0.12128, 1]), np.array([0, 0.01559, 0.12128, 1]), np.array([0, 0.05, 0.07938, 1]), 
              np.array([0, 0, 0.10946, 1]), np.array([0, 0.05, 0.07938, 1]), np.array([0, 0, 0, 1])]
elif case == 3:
    start = np.array([-0, -pi/2, 0, 0, 0, 0])
    end = np.array([0, -pi, 0, -0, -0, 0])
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.50519, 0, 0.12128, 1]), np.array([-0.01559, 0.13687, 0, 1]), np.array([0, 0, 0.10946, 1]), 
            np.array([0, -0.05, 0.07938, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0, 1])]
elif case == 4:
    start = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
    end = np.array([pi/2, -pi/2, pi/2, -pi/2, -pi/2, 0])
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.45, -0.05519, 0.12128, 1]), np.array([0, -0.05519, 0.12128, 1]), np.array([0, 0, 0.10946, 1]), 
            np.array([0, 0, 0.10946, 1]), np.array([-0.05, 0, 0.07938, 1]), np.array([0, 0, 0, 1])]
traj = generate_traj_time(2, 201, start, end)
angle = np.zeros((6, 201))
velocity = np.zeros((6, 201))
accel = np.zeros((6, 201))
mass_vec = np.zeros(6)
point_tuple_def = [('Point Coordinate', np.ndarray), ('Segment','i2')] # list that contains the energy consumption and the corresponding row index from assble_qdd
point_tuple = np.zeros((7), dtype = point_tuple_def)
#points = [np.array([0, -0.13687, 0, 1]), np.array([-0.50519, 0, 0.12128, 1]), np.array([-0.01559, 0.13687, 0, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0, 1])] # points of interest
for i, p in enumerate(points):
    point_tuple[i][0] = p

point_tuple[0][1] = 0
point_tuple[1][1] = 1
point_tuple[2][1] = 1
point_tuple[3][1] = 2
point_tuple[4][1] = 3
point_tuple[5][1] = 4
point_tuple[6][1] = 5

for j in range(6):

    angle[j, :] = traj[j].q
    velocity[j, :] = traj[j].qd
    accel[j, :] = traj[j].qdd
    mass_vec[j] = Yu[j].m

mass = np.linalg.norm(mass_vec, 1) # total mass of robot

def unit_vector(q, point, joint_num, t, qd): # angle and velocity matrix should have the size of (201, 6), point = np.array([x, y, z, 1]), t: traj time

    Yu = rtb.models.DH.Yu() # import Yu with all relevant geometric and mechanical data
    #traj = generate_traj_time(2.5, 201)

    trafo_p = np.eye(4)
    trafo_p[:, -1] = point
    joint_jacobian = np.zeros((6, 6))   
    pose_list = []
    poses = Yu.fkine_all(q)
    #print(poses)

    for pose in (poses):
        pose_list.append(np.array(pose))  

    rf_p =  pose_list[joint_num+1] @ trafo_p @ np.array([0, 0, 0, 1]) # Position vector of p in RF0

    for count in range(joint_num+1):
        translation = rf_p - pose_list[count] @ np.array([0, 0, 0, 1])
        translation = translation[:3]
        rotation = pose_list[count] @ np.array([0, 0, 1, 0])
        rotation = rotation[:3]
        joint_jacobian[:3, count] = np.cross(rotation, translation)
        joint_jacobian[3:6, count] = rotation

    lin_ang = joint_jacobian @ qd # velocity[t, :joint_num+1]
    lin_vel = lin_ang[:3]
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
    T_0e = T_0e.A
    R_0e = T_0e[:3, :3]
    R_e0 = np.transpose(R_0e)
    #print(T_e0)
    r6p_0 = r6p_0[:3]
    r6p_0 = R_e0 @ r6p_0
    #print(r6p_0)
    trafo_6p = np.eye(4)
    trafo_6p[:3, -1] = r6p_0[:3]
    #print(trafo_6p)
    Yu.tool = trafo_6p
    jacobian_p = Yu.jacob0(q)
    joint_jacobian = jacobian_p

    M_q = Yu.inertia(q)
    Aq_inv = joint_jacobian @ np.linalg.inv(M_q) @ np.transpose(joint_jacobian)

    if all(element == 0 for element in vel_u):
        m_refl = 0
        #print('oh no')
    else:
        m_refl = 1/(np.transpose(vel_u) @ Aq_inv[:3, :3] @ vel_u)


    return vel_u, m_refl, lin_vel, rf_p

m_refl = np.zeros((7, 201))
lin_vel = np.zeros((7, 3, 201))
rf_p_storage = np.zeros((7, 3, 201))

for t in range(201):
    #if t == 1:
    #    print(q)
    q = angle[:, t]
    qd = velocity[:, t]

    for i, point_list in enumerate(point_tuple):
        #print(f'r = {r}\n')
        result = unit_vector(q, point_list[0], point_list[1], t, qd)
        lin_vel[i, :, t] = result[2]
        m_refl[i, t] = result[1]
        rf_p_storage[i, :, t] = result[3][:3]

n = 6
print(f'Reflected mass: \n{m_refl[n, :]}')
#print(lin_vel[n, :, :])
vel_abs = np.linalg.norm(lin_vel[n, :, :], axis=0)
#print(vel_abs)
refl_max = np.argmax(m_refl[n, :])
#print(refl_max)
#print(m_refl[n, refl_max])
momentum = np.multiply(vel_abs, m_refl[n, :])
print(f'Momentum trajectory: \n{momentum}')
ax = plt.axes(111, projection='3d')

lin_vel_plot = lin_vel[n, :, :]
start = np.transpose(rf_p_storage[n, :, :])
#print(rf_p_storage[n, :, :])
xline2 = start[:, 0]
yline2 = start[:, 1]
zline2 = start[:, 2]


ax = plt.axes(111, projection='3d')
#ax.set_xlim([-8, 8])
#ax.set_ylim([-8, 8])
#ax.set_zlim([-1, 1])
#ax.set_xlim(0, 1)
#ax.set_ylim(0, 1)
#ax.set_zlim(0, 1)
#ax.set_aspect('equal', adjustable='box')
ax.axis('scaled')
ax.set_xlabel('x coordinate in m')
ax.set_ylabel('Y coordinate in m')
ax.set_zlabel('Z coordinate in m')
#ax.plot3D(xline, yline, zline)
#ax.plot3D(xline1, yline1, zline1, color='blue')
ax.plot3D(xline2, yline2, zline2, color='red', linewidth=1, label='Trajectory of center of mass')
"""
for i in range(201):
#for i in range(5):
    if i == 0:
        ax.quiver(start[i, 0], start[i, 1], start[i, 2], lin_vel_plot[0, i], lin_vel_plot[1, i], lin_vel_plot[2, i], length=0.4*np.linalg.norm((start[i, :]-lin_vel_plot[:, i]), 2), arrow_length_ratio=0.05, normalize='false', color='green', label='Momentum vector of robot on its center of mass')
    elif i % 4 == 0:
    #ax.quiver(start[i, 0], start[i, 1], start[i, 2], lin_max[i, 0], lin_max[i, 1], lin_max[i, 2], arrow_length_ratio=0.01, length=np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True')
        ax.quiver(start[i, 0], start[i, 1], start[i, 2], lin_vel_plot[0, i], lin_vel_plot[1, i], lin_vel_plot[2, i], length=0.4*np.linalg.norm((start[i, :]-lin_vel_plot[:, i]), 2), arrow_length_ratio=0.05, normalize='false', color='green')#length=0.01*np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True', arrow_length_ratio=0.05, color='blue')
    #ax.quiver(T_cm[i, 0], T_cm[i, 1], T_cm[i, 2], lin_max[i, 0], lin_max[i, 1], lin_max[i, 2], arrow_length_ratio=0.01, length=np.linalg.norm((start[i, :]-lin_max[i, :]), 2), normalize='True')
#plt.show()
#ax.legend()
"""
plt.show()

plt.plot(traj[6], momentum)
plt.show()