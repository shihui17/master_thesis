import roboticstoolbox as rtb
from swift import Swift
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import tools as tls

puma = rtb.models.DH.Puma560()
traj = rtb.jtraj(puma.qz, puma.qr, 100)
#print(np.round(traj.q, 2))
#puma.plot(traj.q)
rtb.xplot(traj.q, block=True)