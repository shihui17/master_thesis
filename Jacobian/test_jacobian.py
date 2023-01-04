import roboticstoolbox as rtb
from math import pi, sin, cos, tan
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH, RevoluteMDH, PrismaticMDH, models
import importlib
import spatialmath.base as base
import spatialmath as sm
from spatialmath.base import getvector
from timeit import default_timer as timer
from ansitable import ANSITable
from swift import Swift
import spatialgeometry as sg
from typing import Tuple
from test_j import *
from jac_compare import assemble


# C:\Users\ShihuiLiu\voraus robotik GmbH\yr Students - Masterarbeit\Codes\matlab2python> 

x = np.zeros(42)

q1 = 30*pi/180
q2 = 30*pi/180
q3 = 30*pi/180
q4 = 30*pi/180
q5 = 30*pi/180
q6 = 30*pi/180
a2 = 450
a3 = 407.8
d2 = 179.1
d3 = -175.1
d4 = -142.2
d5 = -142.2
d6 = -150.7
T_n_TCP_1_4 = 5
T_n_TCP_2_4 = 3
T_n_TCP_3_4 = 2

x[0] = q1
x[1] = q2
x[2] = q3
x[3] = q4
x[4] = q5
x[5] = q6
x[19] = a2
x[20] = a3
x[13] = d2
x[14] = d3
x[15] = d4
x[16] = d5
x[17] = d6
x[39] = T_n_TCP_1_4
x[40] = T_n_TCP_2_4
x[41] = T_n_TCP_3_4

JTool = calcJacobianTCP(x)
print(f'The jacobian matrix calculated by matlab function is: \n {np.round((JTool), 2)}\n')

myrobot = rtb.models.DH.Yu()

i = 0
if i == 0:
    myrobot.tool = np.array([[cos(pi/3), 0, sin(pi/3), 5],
                        [0, 1, 0, 3],
                        [-sin(pi/3), 0, cos(pi/3), 2],
                        [0, 0, 0, 1]]
    )
else:
    myrobot.tool = np.array([[1, 0, 0, 5],
                        [0, 1, 0, 3],
                        [0, 0, 1, 2],
                        [0, 0, 0, 1]]
    ) 

print(myrobot.tool)
q = [pi/6, pi/6, pi/6, pi/6, pi/6, pi/6]       
J = np.round(myrobot.jacob0(q), 2)
print(f'The jacobian matrix calculated by roboticstoolbox in python is: \n {J}\n')
