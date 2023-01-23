import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from timeit import default_timer as timer
from ansitable import ANSITable
from swift import Swift
import math
import spatialgeometry as sg
from typing import Tuple

panda = rtb.models.DH.Panda()


env = Swift()
env.launch(realtime = True)
env.add(panda)


ee_axes = sg.Axes(0.1)
goal_axes = sg.Axes(0.1)
env.add(ee_axes)
env.add(goal_axes)

ee_axes.T = panda.fkine(panda.q)
goal_axes.T = sm.SE3.Trans(0.5, 0.0, 0.5)
env.step(0)

panda.q = panda.qr
env.step(0)
panda.qd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

for _ in range(100):
    env.step(0.05)

# Moore-Penrose
# desired end-effector speed:
ev = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Jacobian in qr config
J = panda.jacob0(panda.qr)
# Moore-Penrose function in numpy
dq = np.linalg.pinv(J) @ ev
print(np.round(dq, 4))
"""
"""
panda.q = panda.qr
env.step(0)
ev = [0.0, 0.0, 0.0, 0.5, 0.0, 0.5]
dt = 0.05
results_q = np.zeros([50, 7])
results_qd = np.zeros([50, 7])
results_qdd = np.zeros([50, 7])

for t in range(50):
    J = panda.jacob0(panda.q)
    J_pinv = np.linalg.pinv(J)
    panda.qd = J_pinv @ ev
    results_q[t, :] = panda.q
    results_qd[t, :] = panda.qd
    results_qdd[t, :] = panda.qdd
    env.step(dt)

print(np.round(results_qdd, 2))
rtb.xplot(results_qd[:, 0], block=True) 

