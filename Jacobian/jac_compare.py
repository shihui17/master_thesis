import roboticstoolbox as rtb
from math import pi, cos, sin
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH, RevoluteMDH, PrismaticMDH, models
import importlib
from spatialmath import base as smb
import spatialmath as sm
from spatialmath.base import getvector
from timeit import default_timer as timer
from ansitable import ANSITable
from swift import Swift
import spatialgeometry as sg
from typing import Tuple
from scipy.spatial.transform import Rotation as R

Yu = rtb.models.DH.Yu()

def assemble (trans, rot):
    temp = np.zeros((4, 4))
    temp[:3, :3] = rot
    temp[:, 3] = trans
    return temp

trans = np.array([5, 3, 2, 1])
q = [pi/6, pi/6, pi/6, pi/6, pi/6, pi/6]

i = 0 

if i == 1:
    for deg in range(0, 91 ,10):
        axis = 'y'
        r_z = R.from_euler(axis, deg, degrees = True)
        r_z_m = r_z.as_matrix()
        T = assemble(trans, r_z_m)
        Yu.tool = T
        J = np.round(Yu.jacob0(q), 2)
        #Je = np.round(Yu.jacobe(q), 2)
        #fwd = Yu.fkine(q)
        #fwd = fwd.A
        #print(fwd[:, 2])
        #print(Je)
        print(f'#The tool is rotated around the {axis} axis by {deg} degrees.\n\nThe transformation matrix from end effector to tool is:\n{Yu.tool}\nThe Jacobian is:\n{J}\n\n--------------------\n')
else:
    for deg in range(0, 91, 10):
        Yu.tool = np.array([[cos(deg*pi/180), 0, sin(deg*pi/180), 5],
                        [0, 1, 0, 3],
                        [-sin(deg*pi/180), 0, cos(deg*pi/180), 2],
                        [0, 0, 0, 1]])
        J = np.round(Yu.jacob0(q), 2)
        #Je = np.round(Yu.jacobe(q), 2)
        #fwd = Yu.fkine(q)
        #fwd = fwd.A
        #print(fwd[:, 2])
        #print(Je)
        print(f'The tool is rotated around the y axis by {deg} degrees.\n\nThe transformation matrix from end effector to tool is:\n{Yu.tool}\nThe Jacobian is:\n{J}\n\n--------------------\n')
    
