import roboticstoolbox as rtb
import numpy as np
import math
from math import pi
from roboticstoolbox import tools as tools
import spatialmath as sm
import spatialgeometry as sg
import matplotlib as mpl
import matplotlib.pyplot as plt
from call_tau import *

def plot_trajectory(rdr):
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained',)
    ax1.plot(rdr[6], np.round(rdr[0].q, 6), label='q1')
    ax1.plot(rdr[6], np.round(rdr[1].q, 6), label='q2')
    ax1.plot(rdr[6], np.round(rdr[2].q, 6), label='q3')
    ax1.plot(rdr[6], np.round(rdr[3].q, 6), label='q4')
    ax1.plot(rdr[6], np.round(rdr[4].q, 6), label='q5')
    ax1.plot(rdr[6], np.round(rdr[5].q, 6), label='q6')
    ax1.set_xlabel('Travel Time in s')
    ax1.set_ylabel('joint angle in rad')
    ax1.legend()

    ax2.plot(rdr[6], np.round(rdr[0].qd, 6), label='q1')
    ax2.plot(rdr[6], np.round(rdr[1].qd, 6), label='q2')
    ax2.plot(rdr[6], np.round(rdr[2].qd, 6), label='q3')
    ax2.plot(rdr[6], np.round(rdr[3].qd, 6), label='q4')
    ax2.plot(rdr[6], np.round(rdr[4].qd, 6), label='q5')
    ax2.plot(rdr[6], np.round(rdr[5].qd, 6), label='q6')
    ax2.set_xlabel('Travel Time in s')
    ax2.set_ylabel('joint velocity in 1/s')
    ax2.legend()

    ax3.plot(rdr[6], np.round(rdr[0].qdd, 6), label='q1')
    ax3.plot(rdr[6], np.round(rdr[1].qdd, 6), label='q2')
    ax3.plot(rdr[6], np.round(rdr[2].qdd, 6), label='q3')
    ax3.plot(rdr[6], np.round(rdr[3].qdd, 6), label='q4')
    ax3.plot(rdr[6], np.round(rdr[4].qdd, 6), label='q5')
    ax3.plot(rdr[6], np.round(rdr[5].qdd, 6), label='q6')
    ax3.set_xlabel('Travel Time in s')
    ax3.set_ylabel('joint acceleration in $1/s^2$')
    ax3.set_ylim(bottom=-5, top=5)
    ax3.legend()
    plt.show()