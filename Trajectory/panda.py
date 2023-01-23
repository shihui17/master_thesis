import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from timeit import default_timer as timer
from ansitable import ANSITable
from swift import Swift
import math
from math import pi
import spatialgeometry as sg
from typing import Tuple

panda = rtb.models.Panda()
print(panda)
env = Swift()
env.launch(realtime = True)
env.add(panda)