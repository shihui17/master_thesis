import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, ERobot, RevoluteDH
from spatialmath import SE3
from spatialmath.base import argcheck
from call_tau import *

Yu = rtb.models.DH.Yu()

q = [1, 0, 2, 0.5, 0, 3]
qd = [0.2, 2, 2, 0.2, 0.2, 0.2]
qdd = [0, 0.1, 0.2, 0, 0.3, 0]

tau1 = cal_tau(q, qd, qdd)
tau2 = Yu.rne(q, qd, qdd)
print(f'{tau1}\n\n{tau2}')