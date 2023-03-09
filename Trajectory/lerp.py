import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def lerp_func(x1, x2, y1, y2, x):
    k = (y2 - y1)/(x2 - x1)
    #print(k)
    return k * (x - x1) + y1

def lerp_func_integral(x1, x2, y1, y2, x, qd0):
    k = (y2 - y1)/(x2 - x1)
    return (0.5 * k * x**2 - k * x1 * x + y1 * x - (0.5 * k * x1**2 - k * x1 * x1 + y1 * x1)) + qd0

def lerp_func_double_integral(x1, x2, y1, y2, x, qd0, q0):
    k = (y2 - y1)/(x2 - x1)
    a2 = qd0 - 0.5 * k * x1**2 + k * x1 * x1 - y1 * x1
    return 1/6 * k * x**3 - 0.5 * k * x1 * x**2 + 0.5 * y1 * x**2 + a2 * x - (1/6 * k * x1**3 - 0.5 * k * x1 * x1**2 + 0.5 * y1 * x1**2 + a2 * x1) + q0

x1 = 0.4
x2 = 0.8
y1 = 2.88947575
y2 = -0.06607621
#print(lerp_func_double_integral(x1, x2, y1, y2, 1.2, 1.73162368, 2.07561222))

t_val = np.linspace(x1, x2, 10)

qdd_sample = np.zeros(10)
qdd_sample = [lerp_func(x1, x2, y1, y2, t) for t in t_val]
#print(qdd_sample)
#print(lerp_func_integral(x1, x2, y1, y2, 0.8, 1.16694377))
qd_sample = [lerp_func_integral(x1, x2, y1, y2, t, 1.16694377) for t in t_val]
#print(qd_sample)

#plt.plot(t_val, qd_sample)
#plt.xlim(left = 0)
##plt.ylim(bottom = 0)
#plt.show()

#print(lerp_func_double_integral(x1, x2, y1, y2, 0.8, 1.16694377, 0.75773109))
#print(integrate.simpson(qd_sample, t_val)+0.75773109)
#print(integrate.simpson(qdd_sample, t_val)+1.16694377)

#k = (y2 - y1)/(x2 - x1)
#print(f'k = {1/6 * k}')
#print(-0.5*k*x1+0.5*y1)