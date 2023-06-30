import numpy as np
import matplotlib.pyplot as plt
from math import pi
from traj import *
"""
start1 = np.array([-pi, -pi/2, pi/2, -pi/2, -pi/2, 0])
end1 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])
trajectory_data = generate_traj_time(0.8, 201, start1, end1)

qd = np.loadtxt("force_result_qd.txt")
time = np.linspace(0, 1.334, 201)
for j in range(6):
    plt.plot(time, qd[j, :])
plt.show()
"""

a = np.arange(16).reshape(4, 4)
b = np.amax(a, axis=1)
c = np.argmax(b)
print(c)