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
from scipy.stats import truncnorm
from scipy.interpolate import interp1d

qdd2 = np.zeros(50)
for i in range(50):
    mu = 0
    lb = (-3 - mu) / 2
    ub = (3 - mu) / 2
    s_qdd_trunc = truncnorm(lb, ub, loc=mu, scale = 2)
    qdd2[i] = s_qdd_trunc.rvs(size=1)
print(qdd2)