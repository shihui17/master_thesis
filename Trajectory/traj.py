'''
@author: Shihui Liu

The function generate_traj_time() generates robot trajectories with trapezoidal velocity profile in joint space using tools.trapezoidal() from roboticstoolbox
Returns joint trajectory data from tg1 to tg6, with relevant attributes like q, qd, qdd. Can later be called by, for example, tg1.q, tg2.qd, tg3.qdd, etc.

'''

import numpy as np
from math import pi
from roboticstoolbox import tools as tools

def generate_traj_time(traj_time, step, start_config, end_config):
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
    t = np.linspace(0, traj_time, step)
    tg1 = tools.trapezoidal(start_config[0], end_config[0], t)
    tg2 = tools.trapezoidal(start_config[1], end_config[1], t)
    tg3 = tools.trapezoidal(start_config[2], end_config[2], t)
    tg4 = tools.trapezoidal(start_config[3], end_config[3], t)
    tg5 = tools.trapezoidal(start_config[4], end_config[4], t)
    tg6 = tools.trapezoidal(start_config[5], end_config[5], t)

    return tg1, tg2, tg3, tg4, tg5, tg6, t, step