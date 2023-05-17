from math import pi
import roboticstoolbox as rtb
import numpy as np
from traj import generate_traj_time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

Yu = rtb.models.DH.Yu()

traj = generate_traj_time(2.5, 201)
angle = np.zeros((6, 201))
velocity = np.zeros((6, 201))
accel = np.zeros((6, 201))
mass_vec = np.zeros(6)

for j in range(6):

    angle[j, :] = traj[j].q
    velocity[j, :] = traj[j].qd
    accel[j, :] = traj[j].qdd
    mass_vec[j] = Yu[j].m

mass = np.linalg.norm(mass_vec, 1)
# define a point on the robot
r1 = np.array([-0.450, 0, 0.13687, 1]) # Position vector of p1 relative to RF2 (segment 2)

q = angle[:, 75]
fwd = Yu.fkine_all(q)
fwd_j2 = np.array(fwd[2])
trafo_p1 = np.eye(4)
trafo_p1[:, -1] = r1
rf_p1 = fwd_j2 @ trafo_p1
r1_0 = rf_p1[:, -1] # Position vector of p1 in RF0 (robot base)
fwd_ee = np.array(fwd[-1])
r6_0 = fwd_ee[:, -1]
r61_0 = r1_0 - r6_0 # Position vector from RF6 to p1 in RF0
print(r61_0)
trafo_61 = np.eye(4)
trafo_61[:3, -1] = r61_0[:3]
print(trafo_61)

#Yu.tool = trafo_61

qd = velocity[:, 0]
jacobian_p1 = Yu.jacob0(q)
print(np.round(jacobian_p1, 3))

M_q = Yu.inertia(q)

Aq_inv = jacobian_p1 @ np.linalg.inv(M_q) @ np.transpose(jacobian_p1)
#print(Aq_inv)

u = np.array([0, 1, 0])
m_refl = 1/(np.transpose(u) @ Aq_inv[:3, :3] @ u)
print(f'reflected mass at point: {m_refl} kg')
print(f'mass of the whole robot: {mass} kg')