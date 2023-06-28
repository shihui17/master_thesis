'''
@author: Shihui Liu

The function generate_traj_time() generates robot trajectories with trapezoidal velocity profile in joint space using tools.trapezoidal() from roboticstoolbox
Returns joint trajectory data from tg1 to tg6, with relevant attributes like q, qd, qdd. Can later be called by, for example, tg1.q, tg2.qd, tg3.qdd, etc.

'''

import numpy as np
from roboticstoolbox import tools as tools

def generate_traj_time(traj_time, step, start_config, end_config, V=None):
    """
    Generate trajectories in joint space with trapezoidal velocity profile with parabolic blend

    :param traj_time: trajectory time in second
    :param step: total time step
    :param start_config: start joint configuration. 1x6 numpy array
    :param end_config: end joint configuration. 1x6 numpy array
    
    :return:
    :tg1, tg2, tg3, tg4, tg5, tg6: joint trajectory data, including joint angle q, velocity qd, acceleration qdd
    :t: time array
    :step: number of time steps
    """
    q_matrix = np.zeros((6, step))
    qd_matrix = np.zeros((6, step))
    qdd_matrix = np.zeros((6, step))
    t = np.linspace(0, traj_time, step)
    tg1 = tools.trapezoidal(start_config[0], end_config[0], t, V)
    tg2 = tools.trapezoidal(start_config[1], end_config[1], t)
    tg3 = tools.trapezoidal(start_config[2], end_config[2], t)
    tg4 = tools.trapezoidal(start_config[3], end_config[3], t)
    tg5 = tools.trapezoidal(start_config[4], end_config[4], t)
    tg6 = tools.trapezoidal(start_config[5], end_config[5], t)

    q_matrix[0, :] = tg1.q
    q_matrix[1, :] = tg2.q
    q_matrix[2, :] = tg3.q
    q_matrix[3, :] = tg4.q
    q_matrix[4, :] = tg5.q
    q_matrix[5, :] = tg6.q

    qd_matrix[0, :] = tg1.qd
    qd_matrix[1, :] = tg2.qd
    qd_matrix[2, :] = tg3.qd
    qd_matrix[3, :] = tg4.qd
    qd_matrix[4, :] = tg5.qd
    qd_matrix[5, :] = tg6.qd

    qdd_matrix[0, :] = tg1.qdd
    qdd_matrix[1, :] = tg2.qdd
    qdd_matrix[2, :] = tg3.qdd
    qdd_matrix[3, :] = tg4.qdd
    qdd_matrix[4, :] = tg5.qdd
    qdd_matrix[5, :] = tg6.qdd   


    np.savetxt("q_orig.txt", q_matrix)
    np.savetxt("qd_orig.txt", qd_matrix)
    np.savetxt("qdd_orig.txt", qdd_matrix)
    
    return tg1, tg2, tg3, tg4, tg5, tg6, t, step