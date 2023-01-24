import ctypes

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

a = calc_tau(0.524, 0.175, 0.8, 0.524, 0.175, 0.8, 0.524, 0.175, 0.8, 0.524, 0.175, 0.8, 0.524, 0.175, 0.8, 0.524, 0.175, 0.8)
print(a.tau1, a.tau2, a.tau3, a.tau4, a.tau5, a.tau6)





