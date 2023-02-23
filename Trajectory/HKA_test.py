import roboticstoolbox as rtb
import numpy as np
import math
from math import pi
from roboticstoolbox import tools as tools
import spatialmath as sm
import spatialgeometry as sg
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import matlib
from call_tau import *
from traj import *
from scipy.interpolate import CubicHermiteSpline, BPoly   
from HKA_Alg import *

result_assemble = heuristic_kalman(200, 20, np.array([[0, 0, 0, 0, 0, 0], [0.0003, 5, 6, 8, 20, 10]]), 0, 0, 6, 9, 30)
result_q = result_assemble[0]
result_qd = result_assemble[1]
result_qdd = result_assemble[2]
time = result_assemble[3]
og_data = result_assemble[4]

print(f'Test!: result_q = {result_q}\n\n result_qd = {result_qd}\n\n result_qdd = {result_qdd}\n\n time_vec = {time}\n\n og_data = {og_data}')

# To do:
# result_q _qd and _qdd should include start point and end point (add one row at start and one row at end)