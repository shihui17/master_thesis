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
            I = [0.04641, 0.03619, 0.02264, 0.00002, -0.00001, 0.00001], # intertia tensor in the oder Ixx, Iyy, Izz, Ixy, Iyz, Ixz
            m = 5.976, # mass
            r = [0.00002, 0.00635, 0.03342] # vector for centre of mass

        )
        # Link 2
        L1 = RevoluteDH(
            d = 179.1, a = 450, alpha = 0,
            I = [0.04223, 0.99637, 0.97760, 0.00006, 0.01707, 0],
            m = 9.572,
            r = [-0.22326, 0, -0.00483]
        )
        # Link 3
        L2 = RevoluteDH(
            d = -175.1, a = 407.8, alpha = 0,
            I = [0.03218, 0.44258, 0.41879, -0.00002, -0.07558, 0],
            m = 4.557,
            r = [-0.18689, -0.00004, 0.02756]
        )
        # Link 4
        L3 = RevoluteDH(
            d = -142.2, a = 0, alpha = pi/2,
            I = [0.01372, 0.00681, 0.01059, 0, -0.00001, 0.00004],
            m = 2.588, 
            r = [-0.00005, 0.02389, 0.00075]
        )
        # Link 5
        L4 = RevoluteDH(
            d = -142.2, a = 0, alpha = -pi/2,
            I = [0.01372, 0.00681, 0.01059, 0.00001, 0.00001, -0.00004],
            m = 2.588,
            r = [0.00008, -0.01619, 0.00675]
        )
        # Link 6
        L5 = RevoluteDH(
            d = -150.7, a = 0, alpha = pi,
            I = [0.01057, 0.00971, 0.00221, -0.00001, 0.00001, -0.00003],
            m = 1.025,
            r = [-0.00013, 0.0015, -0.07384]
        )
        super().__init__(
            [L0, L1, L2, L3, L4 ,L5],
            name = "Yu"
        )

        self._qz = np.array([0, 0, 0, 0, 0, 0])

    @property
    def qz(self):
        return self._qz


#if __name__ == '__main__':
    #robot = Yu()
    #print(robot)