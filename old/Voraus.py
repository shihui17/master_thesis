#!/usr/bin/env python
from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH, RevoluteMDH, PrismaticMDH

class Yu(DHRobot):
    def __init__(self):
        deg = pi/180
        # mass in kg, distance in m

        # Link 1
        L0 = RevoluteDH(
            d = 174.95, # DH Param
            a = 0, # DH Param
            alpha = -pi/2, #DH Param
            I = [4.641e-2, 3.619e-2, 2.264e-2, 0.002e-2, -0.001e-2, 0.001e-2], # intertia tensor in the oder Ixx, Iyy, Izz, Ixy, Iyz, Ixz
            m = 5.976, # mass
            r = [0.02e-3, 6.35e-3, 33.42e-3] # vector for centre of mass

        )
        # Link 2
        L1 = RevoluteDH(
            d = 179.1, a = 450, alpha = 0,
            I = [4.223e-2, 99.637e-2, 97.760e-2, 0.006e-2, 1.707e-2, 0],
            m = 9.572,
            r = [-223.26e-3, 0, -4.83e-3]
        )
        # Link 3
        L2 = RevoluteDH(
            d = -175.1, a = 407.8, alpha = 0,
            I = [3.218e-2, 44.258e-2, 41.879e-2, -0.002e-2, -7.558e-2, 0],
            m = 4.557,
            r = [-186.89e-3, -0.04e-3, 27.56e-3]
        )
        # Link 4
        L3 = RevoluteDH(
            d = -142.2, a = 0, alpha = pi/2,
            I = [1.372e-2, 0.681e-2, 1.059e-2, 0, -0.001e-2, 0.004e-2],
            m = 2.588, 
            r = [-0.05e-3, 23.89e-3, 0.75e-3]
        )
        # Link 5
        L4 = RevoluteDH(
            d = -142.2, a = 0, alpha = -pi/2,
            I = [1.372e-2, 0.681e-2, 1.059e-2, 0.001e-2, 0.001e-2, -0.004e-2],
            m = 2.588,
            r = [0.08e-3, -16.19e-3, 6.75e-3]
        )
        # Link 6
        L5 = RevoluteDH(
            d = -150.7, a = 0, alpha = pi,
            I = [1.057e-2, 0.971e-2, 0.221e-2, -0.001e-2, 0.001e-2, -0.003e-2],
            m = 1.025,
            r = [-0.13e-3, 1.5e-3, -73.84e-3]
        )
        super().__init__(
            [L0, L1, L2, L3, L4 ,L5],
            name = "Yu"
        )

        self._qz = np.array[0, 0, 0, 0, 0, 0]
    @property
    def qz(self):
        return self._qz


#if __name__ == '__main__':
    #robot = Yu()
    #print(robot)