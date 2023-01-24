import ctypes
import numpy as np

def cal_tau(q1, dq1, ddq1, q2, dq2, ddq2, q3, dq3, ddq3, q4, dq4, ddq4, q5, dq5, ddq5, q6, dq6, ddq6):
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

    calc_tau = clibrary.tau

    calc_tau.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, 
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, 
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    calc_tau.restype = torque_struct

    a = calc_tau(q1, dq1, ddq1, q2, dq2, ddq2, q3, dq3, ddq3, q4, dq4, ddq4, q5, dq5, ddq5, q6, dq6, ddq6)
    
    tau = np.array([a.tau1, a.tau2, a.tau3, a.tau4, a.tau5, a.tau6])

    return tau





