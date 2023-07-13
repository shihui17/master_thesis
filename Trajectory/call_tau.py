'''
@author: Shihui Liu

Embedding inverse dynamics (.c file) into python script using ctypes
'''

import ctypes
import numpy as np

def cal_tau(array_q, array_dq, array_ddq):
    clibrary = ctypes.CDLL("C:/Codes/master_thesis/Trajectory/tau_lib.so")

    class torque_struct(ctypes.Structure):
        _fields_ = [
            ("tau1", ctypes.c_double),
            ("tau2", ctypes.c_double),
            ("tau3", ctypes.c_double),
            ("tau4", ctypes.c_double),
            ("tau5", ctypes.c_double),
            ("tau6", ctypes.c_double),
        ]
    q = np.zeros(6)
    dq = np.zeros(6)
    ddq = np.zeros(6)
    calc_tau = clibrary.tau

    calc_tau.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, 
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, 
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    calc_tau.restype = torque_struct
    for i in range(6):
        q[i] = array_q[i]
        dq[i] = array_dq[i]
        ddq[i] = array_ddq[i]
    a = calc_tau(q[0], dq[0], ddq[0], q[1], dq[1], ddq[1], q[2], dq[2], ddq[2], q[3], dq[3], ddq[3], q[4], dq[4], ddq[4], q[5], dq[5], ddq[5])
    
    tau = np.array([a.tau1, a.tau2, a.tau3, a.tau4, a.tau5, a.tau6])

    return tau





