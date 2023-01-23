import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from timeit import default_timer as timer
from ansitable import ANSITable
from swift import Swift
import math
from math import pi
import spatialgeometry as sg
from typing import Tuple


Yu = rtb.models.DH.Puma560()
env = Swift()
env.launch(realtime = True)
env.add(Yu)
qr = np.array([0, pi/2, pi/3, 0, pi/6, pi/6])
Yu.q = qr
env.step(0)
dt = 0.05
ev = [0.1, 0.1, 0.0, 0.0, 0.0, 0.0]

for _ in range(50):
    J = Yu.jacob0(Yu.q)
    J_inv = np.linalg.inv(J)
    Yu.qd = J_inv @ ev
    env.step(dt)


