import numpy as np
from scipy.interpolate import BPoly
import matplotlib.pyplot as plt

"""
x = np.array([ 3.,  6., 10., 13., 16.])
y = np.array([0.67164152, 1.11120234, 1.93398019, 2.55414771, 2.99545786])
dy = np.array([0.09458249, 0.1935781,  0.20380581, 0.19417821, 0.09551739])

t = np.array([ 3.,  6., 10., 13., 16.])
q_path = np.array([0.67164152, 1.11120234, 1.93398019, 2.55414771, 2.99545786])
qd_path = np.array([0.09458249, 0.1935781,  0.20380581, 0.19417821, 0.09551739])
qdd_path = np.array([8.69452293e-03, 9.58848730e-03, 2.80075542e-03, 4.32528318e-05, -8.31117476e-03, -8.74073843e-03])

ers = interpolate.CubicHermiteSpline(t, q_path, qd_path)

t_eval = np.linspace(0, 20, 100)
y = ers(t_eval)
y_deriv = ers.derivative()(t_eval)

import matplotlib.pyplot as plt

plt.plot(t, q_path, 'o', label = 'data')
plt.plot(t_eval, y, label = 'spline')
plt.plot(t_eval, y_deriv, label = 'derivative')
plt.legend()
plt.show()
"""

t = np.array([ 3.,  6., 10., 13., 16.])
q_path = np.array([0.67164152, 1.11120234, 1.93398019, 2.55414771, 2.99545786])

qd_path = np.array([0.09458249, 0.1935781,  0.20380581, 0.19417821, 0.09551739])

qdd_path = np.array([8.69452293e-03, 9.58848730e-03, 2.80075542e-03, 4.32528318e-05, -8.31117476e-03])

q_assemble = np.vstack((q_path, qd_path, qdd_path))
q_assemble = np.transpose(q_assemble)
print(q_assemble)

func = BPoly.from_derivatives(t, q_assemble)
t_eval = np.linspace(3, 16, 100)
y = func(t_eval)
y_d = func.derivative()(t_eval)
y_dd = func.derivative(2)(t_eval)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(t_eval, y, label = 'Berstein')
ax2.plot(t_eval, y_d, label = 'First Derivative')
ax3.plot(t_eval, y_dd, label = 'Second Derivative')
plt.show()