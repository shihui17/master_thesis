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
from scipy.interpolate import CubicHermiteSpline, BPoly, interp1d
from scipy.stats import truncnorm
from scipy.integrate import quad, simpson
from lerp import *
from plot_traj import *


def kalman_gain(sig, var_post, mu, mu_meas):
    """
    Performs the update step of a Kalman filter (similar to unscented Kalman filter), 
    calculates the Kalman gain based on current state std. deviation (variance), measured variance, current state mean and measured mean,
    returns the new mean and std. deviation vector stored in a tuple

    :param sig: current state std. deviation vector
    :type sig: 1xn numpy array
    :param var_post: measured variance vector
    :type var_post: 1xn numpy array
    :param mu: current state mean vector
    :type mu: 1xn numpy array
    :param mu_meas: measured mean vector
    :type mu_meas: 1xn numpy array

    :return: the updated mean and std. deviation vector
    """
    #print(f'Param check: sig_qdd = {sig}, var_qdd_post = {var_post}, mu_qdd = {mu}, mu_qdd_meas = {mu_meas}\n')
    L = np.divide(np.square(sig), (np.square(sig) + var_post)) # Kalman gain
    mu_post = mu + np.multiply(L, (mu_meas - mu)) # post mean
    P = np.square(sig) - np.multiply(L, np.square(sig)) # post covariance
    s = np.min([1, np.mean(np.sqrt(var_post))])
    a = 0.8 * (s**2 / (s**2 + np.max(P))) # slowdown factor to slow down the convergence, coefficient (in this case 0.6) can vary (see HKA book)
    #print(P)
    #print(var_q_post)
    sig_new = sig + a * (np.sqrt(P) - sig) # set new std. deviation
    mu_new = mu_post # set new mean
    
    return mu_new, sig_new

def heuristic_kalman(N, Nbest, n, sample_num, joint):
    """
    Applies heuristic Kalman algorithm to a given trajectory of n robot joints,
    returns the optimized q, qd, qdd trajectories for all joints

    :param N: number of random values to be generated
    :param Nbest: number of best candidates to consider
    :param alpha: slow-down factor, applies to function: kalman_gain
    :param sd: legacy
    :param n: number of joints, n=6
    :sample_num: number of sample points on the trajectory to be evaluated, excluding start and end point
    :traj_steps: number of time steps with which the trajectory is generated
    
    :return:
    :result_q, result_qd, result_qdd: optimized joint angle, velocity and acceleration, (sample_num+2) x 6 matrix
    :time_vec: time vector for creating graphs, 1 x (sample_num + 2) array, ranging from 0 sec to traj_steps sec
    """

    mu_q = np.zeros(n)
    mu_qd = np.zeros(n)
    mu_qdd = np.zeros(n)
    sig_qdd = np.zeros(n)
    max_iter = 150
    assble_q = np.zeros((N, n))
    assble_qd = np.zeros((N, n))
    assble_qdd = np.zeros((N, n))
    result_q = np.zeros((sample_num+1, 6))
    result_qd = np.zeros((sample_num+1, 6))
    result_qdd = np.zeros((sample_num+1, 6))
    array_q = np.zeros(6)
    array_qd = np.zeros(6)
    array_qdd = np.zeros(6)
    plot_trajectory(joint)
    traj_time = joint[6][-1]
    u = np.zeros(6) # profile identifier, 1 for positive trapeze, -1 for negative trapeze

    for jt in range(6):
        if joint[jt].qdd[0] > 0:
            u[jt] = 1
        else:
            u[jt] = -1

    time_vec = np.zeros(sample_num+2)
    ctr = 0
    dt = joint[6][1] - joint[6][0]
    energy_og_total = 0 # accumulated energy consumption along unoptimized trajectory
    energy_total = 0 # accumulated energy consumption along optimized trajectory
    t_ref = []

    for i in range(n): # write initial angle, velocity and acceleration for all joints in the first rows of the result lists
        result_q[0, i] = joint[i].q[0]
        result_qd[0, i] = joint[i].qd[0]
        result_qdd[0, i] = joint[i].qdd[0]
    #print(result_qdd)

    for sn in range(sample_num): # now iterate through all sample points
        flag = False
        energy_og = 0 # initialize pre-optimization energy consumption
        iter = 0 # initialize iteration counter
        t2 = traj_time/(sample_num+1)*(sn+1) # current sample point
        time_vec[sn+1] = t2 # current sample point
        t1 = time_vec[sn] # previous sample point, we are at sample point sn+1 right now, meaning previous sample point is sn
        q_ref = np.zeros(6) # initialize reference joint angle array, where original joint angles at sample point sn are stored
        ref_index = np.where(np.round(joint[6], 2) == np.round(t2, 2))  

        for ii in range(6): # fill out q_ref with the respective (original) joint angles
            q_ref[ii] = joint[ii].q[ref_index[0]]

        mu_index = math.floor(200/(sample_num+1)*(sn+1))

        while ctr < mu_index: # iterate from current counter to mu_index (the index that corresponds to the current sample point), counter is NOT reset after iteration is done

            for joint_num in range(6): # generate q, qd, qdd arrays for torque calculation with cal_tau()
                array_q[joint_num] = joint[joint_num].q[ctr]
                array_qd[joint_num] = joint[joint_num].qd[ctr]
                array_qdd[joint_num] = joint[joint_num].qdd[ctr]
            
            torq_og = cal_tau(array_q, array_qd, array_qdd)
            power_og = np.multiply(torq_og, array_qd)
            energy_og = energy_og + np.linalg.norm(power_og, 1) * dt
            ctr = ctr + 1

        print(f'Calculation stops at trajectory time {joint[6][ctr]} s\n')
        print(f'Original energy consumption until sample point {sn+1} is: {energy_og} J\n')

        #if sn == sample_num - 1:
        #    energy_og = 0

        energy_og_total = energy_og_total + energy_og

        # initialize mean vector mu_q, mu_qd, mu_qdd, and std vector sig_qdd
        for i in range(n):
            
            mu_q[i] = joint[i].q[mu_index]
            mu_qd[i] = joint[i].qd[mu_index]
            mu_qdd[i] = joint[i].qdd[mu_index]
            #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
            if mu_qdd[i] == 0:
                sig_qdd[i] = 1 # initialize std vector, hard coded for now
                flag = True
            else:
                sig_qdd[i] = abs(0.2*mu_qdd[i])
            
        print(f'Debugging: sigma through initialisation: {sig_qdd[i]}\n')

        test_qd = mu_qd

        #print(f'Original joint angle vector:\n{mu_q}\n')
        print(f'Original joint velocity vector:\n{mu_qd}\n')
        print(f'Original joint acceleration vector:\n{mu_qdd}\n')

        #ref_torq = cal_tau(mu_q, mu_qd, mu_qdd) # original output torque for reference
        #ref_power = np.multiply(ref_torq, mu_qd) # calculate orginal power output of each joint
        #ref_total_power = np.linalg.norm(ref_power, 1) # summation to get total power output
        #print(f'Original total power output: {ref_total_power} W\n')

        while iter <= max_iter: # begin kalman iteration
            #print(f'current sig: {sig_qdd}')
            for i in range(6): # generate gaussian destribution for acceleration of each joint (N random values), results written in assble_qdd (an Nx6 matrix)
                #print(sig_qdd[i])
                
                if mu_qdd[i] < 0 and flag == False:
                    lb = (max(1.5 * mu_qdd[i], -1.5 * abs(result_qdd[0, i])) - mu_qdd[i]) / sig_qdd[i]
                    #print(f'1.5 * mu_qdd[{i}] = {1.5*mu_qdd[i]}; the other is {-1.5 * abs(result_qdd[0, i])}\n')
                    #print(f'lb = {lb}')
                    ub = (min(0.5 * mu_qdd[i], -0.5 * abs(result_qdd[0, i])) - mu_qdd[i]) / sig_qdd[i]
                   # print(ub)
                    s_qdd_trunc = truncnorm(lb, ub, loc=mu_qdd[i], scale=sig_qdd[i])
                    assble_qdd[:, i] = np.transpose(s_qdd_trunc.rvs(size=N))
                    a = 0
                    #print(a)
                elif mu_qdd[i] > 0 and flag == False:
                    lb = (max(0.5 * mu_qdd[i], 0.5 * abs(result_qdd[0, i])) - mu_qdd[i]) / sig_qdd[i]
                    ub = (min(1.5 * mu_qdd[i], 1.5 * abs(result_qdd[0, i])) - mu_qdd[i]) / sig_qdd[i]
                    s_qdd_trunc = truncnorm(lb, ub, loc=mu_qdd[i], scale=sig_qdd[i])
                    assble_qdd[:, i] = np.transpose(s_qdd_trunc.rvs(size=N))
                    a = 1
                    #print(a)
                else:
                    s_qdd = np.random.normal(0, sig_qdd[i], N)
                    assble_qdd[:, i] = np.transpose(s_qdd)
                    a = 2
                
                #s_qdd = np.random.normal(mu_qdd[i], sig_qdd[i], N)
                #assble_qdd[:, i] = np.transpose(s_qdd)
                    #print(a)
            #print(a)
            #print(len(assble_qdd))
            #if sn >= 4 and iter <=3:
            #    print(assble_qdd)
            #print(result_q)

            # Linear interpolation between the previous and the current acceleration

            width = 10
            qdd_sample = np.zeros((6, width))
            qd_sample = np.zeros((6, width))
            q_sample = np.zeros((6, width))


            t_val = np.linspace(t1, t2, num = width)          
            #print(q_ref)
            #print(joint[0].q[28])
            delta_t = t_val[1] - t_val[0]
            power = np.zeros((N, 6))

            energy_val_def = [('Energy','f8'), ('Number','i2')] # list that contains the energy consumption and the corresponding row index from assble_qdd
            energy_val = np.zeros((N), dtype = energy_val_def)
            cost_total_intv_def = [('Regulated Energy', 'f8'), ('Number', 'i2')]
            cost_total_intv = np.zeros((N), dtype = cost_total_intv_def)
  
            for i in range(N): # iterate through all N randomly generated accelerations
                #print(f'Debug: counter check, {i}')

                energy_total_intv = 0 # initialize total energy consumed by robot between time steps t1 and t2 (interval energy)
                coeff = np.zeros(6)
                q_list = []
                qd_compare = []
                #print(N)
                #print(i)
                for j in range(6): # iterate through all joints, j = 0, ..., 5
                    qdd1 = result_qdd[sn, j] # start qdd for lerp
                    qdd2 = assble_qdd[i, j] # end qdd for lerp
                    qd1 = result_qd[sn, j]
                    qd2 = assble_qd[i, j]
                    qdd_sample[j, :] = [lerp_func(t1, t2, qdd1, qdd2, t) for t in t_val]
                    qd_sample[j, :] = [lerp_func_integral(t1, t2, qdd1, qdd2, t, result_qd[sn, j]) for t in t_val] # initial condition: qd0 = result_qd[sn, j], qd from previous time step
                    q_sample[j, :] = [lerp_func_double_integral(t1, t2, qdd1, qdd2, t, result_qd[sn, j], result_q[sn, j]) for t in t_val] # qd0, q0 from result_qd, result_q
                
                cal_tau_qdd = np.transpose(qdd_sample) # qdd matrix for torque calculation, has a dimension of Width x 6
                cal_tau_qd = np.transpose(qd_sample)
                assble_qd[i, :] = cal_tau_qd[-1, :]
                cal_tau_q = np.transpose(q_sample) # first row of q_sample looks like this [q0(t1), q0(t1+dt), q0(t1+2dt), ... , q0(t2)]
                q_opt = q_sample[:, -1]
                delta_q = q_opt - q_ref
                assble_q[i, :] = cal_tau_q[-1, :]

                for k in range(width): # iterate through all samples generated by lerp    
                    torq_vec = cal_tau(cal_tau_q[k, :], cal_tau_qd[k, :], cal_tau_qdd[k, :])
                    velocity_vec = cal_tau_qd[k, :]
                    power_val = abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1)) # element-wise multiplication, then take 1-Norm
                    energy_total_intv = energy_total_intv + power_val * delta_t

                energy_val[i] = (energy_total_intv, i)
                sign_array = np.sign(delta_q) # recall that delta_q = q_opt - q_ref, we want to encourage q_opt > q_ref if qdd[0] > 0, vice versa
                for z in range(6):
                    if z == 1:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -200 if sign_array[z] > 0 else 200
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 200 if sign_array[z] < 0 else -200
                    elif z == 0:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -100 if sign_array[z] > 0 else 100
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 100 if sign_array[z] < 0 else -100
                    elif z == 2:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -50 if sign_array[z] > 0 else 40
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 40 if sign_array[z] < 0 else -50       
                    elif z == 4:                
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -30 if sign_array[z] > 0 else 30
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 30 if sign_array[z] < 0 else -30                         
                    elif z == 5:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = 90 if sign_array[z] > 0 else -90
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = -90 if sign_array[z] < 0 else 90
                    else: 
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -20 if sign_array[z] > 0 else 20
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 20 if sign_array[z] < 0 else -20
                #print(f'debug: penalty term{coeff * (sign_array * delta_q**2)}')
                cost_total_intv[i] = (energy_total_intv + (np.linalg.norm(coeff * (sign_array * delta_q**2), 1))**2, i)
                #print(energy_val[i])
                    #print(f'End evaluation: {k+1}th row of the torque calculation matrices\n')

            sorted_energy_val = np.sort(energy_val, order='Energy')
            sorted_cost_total = np.sort(cost_total_intv, order = 'Regulated Energy')
            post_q = np.zeros((Nbest, 6))
            post_qd = np.zeros((Nbest, 6))
            post_qdd = np.zeros((Nbest, 6))
            num_array = sorted_cost_total['Number']

            for i in range(Nbest):
                num = num_array[i] # returns the index for num-th best candidate
                post_q[i, :] = assble_q[num, :] # place the num th row of the assembled q-matrix onto the i-th row of the post_q matrix
                post_qd[i, :] = assble_qd[num, :]
                post_qdd[i, :] = assble_qdd[num, :]

            mu_qdd_meas = np.mean(post_qdd, 0)
            var_qdd_post = np.var(post_qdd, 0)
            
            new_mu_sig_qdd = kalman_gain(sig_qdd, var_qdd_post, mu_qdd, mu_qdd_meas)
            mu_qdd_old = mu_qdd
            mu_qdd = new_mu_sig_qdd[0]
            sig_qdd = new_mu_sig_qdd[1]

            if all(i < 1e-6 for i in var_qdd_post) == True:
                print(f'exited loop at iter = {iter}')
                break
            iter = iter + 1

        t_ref.append(t_val)
        result_q[sn+1, :] = assble_q[num_array[0], :]      
        print(f'post acceleration matrix post_qdd = {post_qdd}\nwith a variance of {var_qdd_post}\nStd.Dev = {sig_qdd}\n\n')#print(test_qd)
        result_qd[sn+1, :] = assble_qd[num_array[0], :]
        result_qdd[sn+1, :] = assble_qdd[num_array[0], :]
        energy_total = energy_total + energy_val[num_array[0]]['Energy']

        print(f'Energy consumption converges at {assble_qdd[num_array[0], :]} 1/s^2, total energy consumption: {energy_val[num_array[0]]} J\n')

    print(result_q)
    print(result_qd)
    print(result_qdd)
    #print(t_ref)
    print(f'Optimization ended.\nOriginal energy consumption of the given trajectory is: {energy_og_total} J.\nTotal energy consumption of the optimizied trajectory is: {energy_total} J.\n')

    np.savetxt("result_q_int.txt", result_q)
    np.savetxt("result_qd_int.txt", result_qd)
    np.savetxt("result_qdd_int.txt", result_qdd)
    np.savetxt("time_vec_int.txt", time_vec)    

    return result_q, result_qd, result_qdd, time_vec, joint

start = np.array([-pi, -pi, pi/2, -pi/2, -pi/2, 0])
end = np.array([0, -0.749*pi, 0.69*pi, 0.444*pi, -0.8*pi, -pi])
joint = generate_traj_time(2, 201, start, end)
results = heuristic_kalman(40, 4, 6, 6, joint)

joint = results[4]
time = results[3][0:7]
q = results[0]
qd = results[1]
qdd = results[2]
width = 10
qdd_graph = np.zeros((6, width))
qd_graph = np.zeros((6, width))
q_graph = np.zeros((6, width))
print(time)





for j_num in range(6):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
    ax1.plot(joint[6], joint[j_num].q, color='blue')
    ax1.plot(time, q[:, j_num], 'r+')
    ax2.plot(joint[6], joint[j_num].qd, color='blue')
    ax2.plot(time, qd[:, j_num], 'r+')
    ax3.plot(joint[6], joint[j_num].qdd, color='blue')
    ax3.plot(time, qdd[:, j_num], 'r+')
    for a, b in zip(time, np.round(qdd[:, j_num], 3)):
        plt.text(a, b, str(b))

plt.show()