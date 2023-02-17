import roboticstoolbox as rtb
from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH, RevoluteMDH, PrismaticMDH, models
import importlib
from spatialmath import base as smb
import spatialmath as sm
from spatialmath.base import getvector
from timeit import default_timer as timer
from ansitable import ANSITable
from swift import Swift
import math
import spatialgeometry as sg
from typing import Tuple
from scipy.spatial.transform import Rotation as R

myrobot = rtb.models.DH.Yu()
Rot = R.from_matrix('z', 90, degrees = True)

i = 1
if i == 0:
    myrobot.tool = np.array([[0.87, 0, -0.5, 5],
                        [0, 1, 0, 3],
                        [-0.5, 0, 0.87, 2],
                        [0, 0, 0, 1]]
    )
else:
    myrobot.tool = np.array([[1, 0, 0, 5],
                        [0, 1, 0, 3],
                        [0, 0, 1, 2],
                        [0, 0, 0, 1]]
    )

myrobot.rne(q, qd, qdd)


q = [pi/6, pi/6, pi/6, pi/6, pi/6, pi/6]       
fwd = myrobot.fkine(q)
T = fwd.A
"""
n = myrobot.n
L = myrobot.links
J = np.zeros((6, myrobot.n), dtype=q.dtype)  # type: ignore
U = myrobot.tool.A
U = L[5].A(q[5]).A @ U
print(U)
d = np.array(
            [
            -U[0, 0] * U[1, 3] + U[1, 0] * U[0, 3],
            -U[0, 1] * U[1, 3] + U[1, 1] * U[0, 3],
            -U[0, 2] * U[1, 3] + U[1, 2] * U[0, 3],
            ]
            )

delta = U[2, :3]  # nz oz az
print(delta)
J[:, 5] = np.r_[d, delta]
print(J)
"""

"""
for j in range(n - 1, -1, -1):
    U = L[j].A(q[j]).A @ U  # type: ignore
    if not L[j].sigma:
    # revolute axis
        d = np.array(
            [
            -U[0, 0] * U[1, 3] + U[1, 0] * U[0, 3],
            -U[0, 1] * U[1, 3] + U[1, 1] * U[0, 3],
            -U[0, 2] * U[1, 3] + U[1, 2] * U[0, 3],
            ]
            )
        delta = U[2, :3]  # nz oz az
        print(delta)
    else:
    # prismatic axis
        d = U[2, :3]  # nz oz az
        delta = np.zeros((3,))

    J[:, j] = np.r_[d, delta]       

"""

J = np.round(myrobot.jacob0(q), 2)
J1 = np.round(myrobot.jacobe(q), 2)
print(J)
print(J1)

Z = np.zeros((3, 3), dtype=T.dtype)
R = smb.t2r(T)
Tz = np.block([[R, Z], [Z, R]])
J2 = Tz @ J1
print(np.round(J2,2))
"""
fwd = myrobot.fkine(q)
T = fwd.A
R = T[:3, :3]
print(R)

Z = np.zeros((3, 3), dtype=T.dtype)
rotJacobian = np.block([[R, Z], [Z, R]])
print(rotJacobian)
jacob0 = rotJacobian @ J0
print(np.round(jacob0, 2)) """

"""
dq = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
e = J @ dq
print(e)
v_norm = e[0]**2 + e[1]**2 + e[2]**2
print(v_norm)

print(myrobot.links[0]) 
print(myrobot.tool)"""