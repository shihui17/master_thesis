from tau import *
from scipy.integrate import simpson
import numpy as np
from call_tau import *

def calculate_energy(q_torq, qd_torq, qdd_torq, time):
    power_val = []
    for i in range(len(q_torq)):
        torq_vec = cal_tau(q_torq[i, :], qd_torq[i, :], qdd_torq[i, :])
        vel_vec = qd_torq[i, :]
        power_vec = np.linalg.norm(np.multiply(torq_vec, vel_vec), 1)
        power_val.append(power_vec)           
    energy = simpson(power_val, time)
    return energy