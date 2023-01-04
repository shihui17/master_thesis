import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

rx_var = rtb.ET.Rx()
rx_cons = rtb.ET.Rx(np.pi/2)
print(rx_cons)
transform = rx_cons.A()
print(transform)

"""
q = np.pi/3
conv = q*180/np.pi
transform = rx_var.A(q)
sm_transform = sm.SE3(transform)
print(f"Resulting SE3 at {int(conv)} degrees: \n{sm_transform}")
"""
"""
ty_var = rtb.ET.ty()
ty_cons = rtb.ET.ty(0.25)

trans = ty_cons.A()
sm_trans = sm.SE3(trans)
print(f"SE3: \n{sm_trans}")"""

E1 = rtb.ET.tz(0.333)
E2 = rtb.ET.Rz()
E3 = rtb.ET.Ry()
E4 = rtb.ET.tz(0.316)
E5 = rtb.ET.Rz()
E6 = rtb.ET.tx(0.0825)
E7 = rtb.ET.Ry(flip=True)
E8 = rtb.ET.tx(-0.0825)
E9 = rtb.ET.tz(0.384)
E10 = rtb.ET.Rz()
E11 = rtb.ET.Ry(flip=True)
E12 = rtb.ET.tx(0.088)
E13 = rtb.ET.Rx(np.pi)
E14 = rtb.ET.tz(0.107)
E15 = rtb.ET.Rz()

total = rtb.ETS([E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15])

print(total.joints()[4])
print(total[1].jindex)

"""
q = np.array([0, -0.3, 0, -2.2, 0, 2, 0.79])

forward = np.eye(4)

for et in total:
    if et.isjoint:
        forward = forward @ et.A(q[et.jindex])
    else:
        forward = forward @ et.A()

print(sm.SE3(forward)) 

print(f"the fkine method: \n{total.fkine(q)}")  """

