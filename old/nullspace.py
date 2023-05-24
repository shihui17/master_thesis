import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from swift import Swift
import spatialgeometry as sg

env = Swift()

env.launch(realtime=True)
env.set_camera_pose([1.3, 0, 0.4], [0, 0, 0.3])

panda = rtb.models.Panda()
panda.q = panda.qr
env.add(panda)
ee_axes = sg.Axes(0.1)
goal_axes = sg.Axes(0.1)
env.add(ee_axes)
env.add(goal_axes)

def null_project(robot, q, qnull, ev, l):
    J0 = robot.jacob0(q)
    J0_pinv = np.linalg.pinv(J0)
    qd = J0_pinv @ ev + (1.0 / l) * (np.eye(robot.n) - J0_pinv @ J0) @ qnull.reshape(robot.n,)
    return qd