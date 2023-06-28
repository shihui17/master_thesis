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
import time as ti

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
    :param n: number of joints, n=6 for Yu+
    :sample_num: number of sample points on the trajectory to be evaluated, excluding start and end point
    :joint: joint data extracted from joint trajectories
    
    :return:
    :result_q, result_qd, result_qdd: optimized joint angle, velocity and acceleration, (sample_num+2) x 6 matrix
    :time_vec: time vector for creating graphs, 1 x (sample_num + 2) array, ranging from 0 sec to traj_steps sec
    """

    mu_q = np.zeros(n) # mean vector of joint angle
    mu_qd = np.zeros(n) # mean vector of joint velocity
    mu_qdd = np.zeros(n) # mean vector of joint acceleration
    sig_qdd = np.zeros(n) # std.
    max_iter = 150 
    assble_q = np.zeros((N, n)) # matrix containing all generated joint angle elements
    assble_qd = np.zeros((N, n)) # matrix containing all generated joint velocity elements
    assble_qdd = np.zeros((N, n)) # matrix containing all generated joint acceleration elements
    result_q = np.zeros((sample_num+2, 6)) # result matrix, first row is joint data at t=0, last row is joint data at end of trajectory
    result_qd = np.zeros((sample_num+2, 6))
    result_qdd = np.zeros((sample_num+2, 6))
    time_vec = np.zeros(sample_num+2) # the time step associated to the supporting points shown in result matrices
    array_q = np.zeros(6) # best joint angle combination that results in highest energy efficiency
    array_qd = np.zeros(6) # best joint velocity combination
    array_qdd = np.zeros(6) # best joint acceleration combination
    q0 = np.zeros(6) # initial joint angles
    qf = np.zeros(6) # target joint angles
    plot_trajectory(joint) # visualize original trajectories
    traj_time = joint[6][-1] # read trajectory time
    u = np.zeros(6) # profile identifier, 1 for positive trapezoid, -1 for negative trapezoid
    width = 50 # number of increments created for numerical integration
    discrete_q = np.zeros((6, width)) # the discrete joint angle profile for plotting optimal trajectory
    discrete_qd = np.zeros((6, width)) # discrete joint velocity profile
    discrete_qdd = np.zeros((6, width)) # discrete joint acceleration profile
    t_trace = [] # discrete time steps from optimal trajectory
    flag_stationary = np.full(6, False) # flag to signal whether a joint is stationary
    start_time = ti.time() # for computing optimization runtime 
    iter_total = 0 # counter for total number of iterations
    time = joint[6] # original time steps

    for jt in range(6): 
        if joint[jt].qdd[0] > 0: # identify whether positive or negative profile
            u[jt] = 1
        else:
            u[jt] = -1

        q0[jt] = joint[jt].q[0] # identify whether stationary
        qf[jt] = joint[jt].q[-1] # identify whether stationary

        if q0[jt] == qf[jt]: # if stationary, flag[joint] = true
            flag_stationary[jt] = True

    ctr = 0
    dt = joint[6][1] - joint[6][0] # size of time increments
    energy_og_total = 0 # accumulated energy consumption along unoptimized trajectory
    energy_total = 0 # accumulated energy consumption along optimized trajectory
    t_ref = [] # reference time array for plotting & debugging, not always needed

    for i in range(n): # write initial angle, velocity and acceleration for all joints in the first rows of the result lists
        result_q[0, i] = joint[i].q[0]
        result_qd[0, i] = joint[i].qd[0]
        result_qdd[0, i] = joint[i].qdd[0]

    for sn in range(sample_num): # now iterate through all sample points
        flag = False # flag that shows whether acceleration at sample point is zero, default is non-zero
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
            energy_og = energy_og + np.linalg.norm(power_og, 1) * dt # energy = accumulated power over small time increment, yields very similar result (less than 0.5% difference) compared to scipy.simpson
            ctr = ctr + 1

        print(f'Calculation stops at trajectory time {joint[6][ctr]} s\n')
        print(f'Original energy consumption until sample point {sn+1} is: {energy_og} J\n')
        energy_og_total = energy_og_total + energy_og # adds to total energy consumption

        # initialize mean vector mu_q, mu_qd, mu_qdd, and std vector sig_qdd
        for i in range(n):
            mu_q[i] = joint[i].q[mu_index]
            mu_qd[i] = joint[i].qd[mu_index]
            mu_qdd[i] = joint[i].qdd[mu_index]
            #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
            if mu_qdd[i] == 0:
                sig_qdd[i] = 1 # initialize std vector, determined empirically
                flag = True # set acceleration identifier to True
            else:
                sig_qdd[i] = abs(0.2*mu_qdd[i]) # std. deviation is 20% of the original mean acceleration, determined empirically
            
        print(f'Debugging: sigma through initialisation: {sig_qdd[i]}\n')
        test_qd = mu_qd # print for debugging, not needed by optimizer
        print(f'Original joint velocity vector:\n{mu_qd}\n')
        print(f'Original joint acceleration vector:\n{mu_qdd}\n')

        while iter <= max_iter: # begin kalman iteration
            # print(f'current sig: {sig_qdd}')
            for i in range(n): # generate truncated gaussian distribution for acceleration of each joint (N random values), results written in assble_qdd (an Nx6 matrix)
                if flag_stationary[i] == True:
                    assble_qdd[:, i] == 0
                elif mu_qdd[i] < 0: # negative acceleration
                    lb = (max(1.5 * mu_qdd[i], -1.5 * abs(result_qdd[0, i])) - mu_qdd[i]) / sig_qdd[i]
                    ub = (min(0.5 * mu_qdd[i], -0.5 * abs(result_qdd[0, i])) - mu_qdd[i]) / sig_qdd[i]
                    if lb > ub:
                        temp = lb
                        lb = ub
                        ub = temp
                    s_qdd_trunc = truncnorm(lb, ub, loc=mu_qdd[i], scale=sig_qdd[i])
                    #print(f'mu for joint {i} = {mu_qdd[i]}')
                    assble_qdd[:, i] = np.transpose(s_qdd_trunc.rvs(size=N))
                elif mu_qdd[i] > 0: # positive acceleration, swap upper and lower bound
                    lb = (max(0.5 * mu_qdd[i], 0.5 * abs(result_qdd[0, i])) - mu_qdd[i]) / sig_qdd[i]
                    ub = (min(1.5 * mu_qdd[i], 1.5 * abs(result_qdd[0, i])) - mu_qdd[i]) / sig_qdd[i]
                    if lb > ub:
                        temp = lb
                        lb = ub
                        ub = temp
                    s_qdd_trunc = truncnorm(lb, ub, loc=mu_qdd[i], scale=sig_qdd[i])
                    #print(f'mu for joint {i} = {mu_qdd[i]}')
                    assble_qdd[:, i] = np.transpose(s_qdd_trunc.rvs(size=N))
                elif mu_qdd[i] == 0: # acceleration = 0, no truncated distribution needed
                    s_qdd = np.random.normal(0, sig_qdd[i], N)
                    assble_qdd[:, i] = np.transpose(s_qdd)

            # Linear interpolation between the previous and the current acceleration
            width = 10
            qdd_sample = np.zeros((N, 6, width)) # sample acceleration points between previous and current sample point
            qd_sample = np.zeros((N, 6, width)) # sample velocity points
            q_sample = np.zeros((N, 6, width)) # sample angle points
            t_val = np.linspace(t1, t2, num = width) # time array associated to the 3 matrices above          
            delta_t = t_val[1] - t_val[0] # size of time increment
            energy_val_def = [('Energy','f8'), ('Number','i2')] # list that contains the energy consumption and the corresponding row index from assble_qdd
            energy_val = np.zeros((N), dtype = energy_val_def) # list definition
            cost_total_intv_def = [('Regulated Energy', 'f8'), ('Number', 'i2')] # list of total cost, total cost = energy + penalty
            cost_total_intv = np.zeros((N), dtype = cost_total_intv_def) # list definition
  
            for i in range(N): # iterate through all N randomly generated accelerations
                energy_total_intv = 0 # initialize total energy consumed by robot in stage
                coeff = np.zeros(6)
                for j in range(6): # iterate through all joints, j = 0, ..., 5
                    qdd1 = result_qdd[sn, j] # start qdd for lerp, acceleration at the end of last stage
                    qdd2 = assble_qdd[i, j] # end qdd for lerp, acceleration at the end of current stage
                    qd1 = result_qd[sn, j] # start qd, velocity at the end of last stage
                    qd2 = assble_qd[i, j] # end qd, velocity at the end of current stage, print for debugging, not needed in lerp
                    q1 = result_q[sn, j] # start q, angle at the end of last stage
                    qdd_sample[i, j, :] = [lerp_func(t1, t2, qdd1, qdd2, t) for t in t_val]
                    qd_sample[i, j, :] = [lerp_func_integral(t1, t2, qdd1, qdd2, t, qd1) for t in t_val] # initial condition: qd1 = result_qd[sn, j], corresponds to the velocity from previous stage
                    q_sample[i, j, :] = [lerp_func_double_integral(t1, t2, qdd1, qdd2, t, qd1, q1) for t in t_val] # initial condition: qd1, q1, correspond to velocity and angle from previous stage
                
                cal_tau_q = np.transpose(q_sample[i, :, :]) # q matrix for torque calculation, first row of q_sample looks like this [q0(t0), q0(t0+dt), q0(t0+2dt), ... , q0(t1)]
                cal_tau_qdd = np.transpose(qdd_sample[i, :, :]) # qdd matrix for torque calculation, analogous to cal_tau_q
                cal_tau_qd = np.transpose(qd_sample[i, :, :]) # qd matrix for torque calculation, analogous to cal_tau_q
                assble_qd[i, :] = cal_tau_qd[-1, :] # record end-of-stage velocity
                q_opt = q_sample[i, :, -1] # record end-of-stage angles 
                delta_q = q_opt - q_ref # angle difference between optimized angle and original angle for computing penalty function
                #assble_q[i, :] = cal_tau_q[-1, :] # record end-of-stage 
                assble_q[i, :] = q_opt

                for k in range(width): # iterate through all samples generated by lerp    
                    torq_vec = cal_tau(cal_tau_q[k, :], cal_tau_qd[k, :], cal_tau_qdd[k, :])
                    velocity_vec = cal_tau_qd[k, :]
                    power_val = abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1)) # element-wise multiplication, then take 1-Norm
                    energy_total_intv = energy_total_intv + power_val * delta_t # total energy cost in lerp interval

                energy_val[i] = (energy_total_intv, i) # read energy cost with the corresponding sample number i
                sign_array = np.sign(delta_q) # recall that delta_q = q_opt - q_ref, we want to encourage q_opt > q_ref if qdd[0] > 0, vice versa
                for z in range(6):
                    if z == 1:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = 0 if sign_array[z] > 0 else 0
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 0 if sign_array[z] < 0 else -0
                    elif z == 0:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = 0 if sign_array[z] > 0 else 0
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 0 if sign_array[z] < 0 else 0
                    elif z == 2:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = 00 if sign_array[z] > 0 else 00
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 00 if sign_array[z] < 0 else -00       
                    elif z == 4:                
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -0 if sign_array[z] > 0 else 0
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 0 if sign_array[z] < 0 else -0                         
                    elif z == 5:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -0 if sign_array[z] > 0 else 0
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 0 if sign_array[z] < 0 else -0
                    else: 
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -0 if sign_array[z] > 0 else 0
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 0 if sign_array[z] < 0 else -0
                cost_total_intv[i] = (energy_total_intv + (np.linalg.norm(coeff * (sign_array * delta_q**2), 1))**2, i) # compute total cost = energy + penalty

            sorted_cost_total = np.sort(cost_total_intv, order = 'Regulated Energy')
            post_q = np.zeros((Nbest, 6)) # initialize posterior angle matrix from Nbest candidates
            post_qd = np.zeros((Nbest, 6)) # posterior velocity
            post_qdd = np.zeros((Nbest, 6)) # posterior acceleration
            num_array = sorted_cost_total['Number']

            for i in range(Nbest):
                num = num_array[i] # returns the index for num-th best candidate
                post_q[i, :] = assble_q[num, :] # place the num-th row of the assembled q-matrix (that contains all N generated angle vectors) onto the i-th row of the post_q matrix
                post_qd[i, :] = assble_qd[num, :] # same for velocity
                post_qdd[i, :] = assble_qdd[num, :] # same for acceleration

            mu_qdd_meas = np.mean(post_qdd, 0) # compute posterior mean
            var_qdd_post = np.var(post_qdd, 0) # compute posterior variance
            new_mu_sig_qdd = kalman_gain(sig_qdd, var_qdd_post, mu_qdd, mu_qdd_meas) # compute Kalman Gain
            mu_qdd = new_mu_sig_qdd[0] # new mean acceleration vector
            sig_qdd = new_mu_sig_qdd[1] # new standard deviation of acceleration vectors

            id = True
            for var in var_qdd_post:
                if var == 0:
                    continue
                if var > 1e-4:
                    id = False
            if id == True:
                print(f'exited loop at iter = {iter}')
                break
            iter = iter + 1
    
        iter_total = iter_total + iter
        t_ref.append(t_val)
        result_q[sn+1, :] = assble_q[num_array[0], :]      
        print(f'post acceleration matrix post_qdd = {post_qdd}\nwith a variance of {var_qdd_post}\nStd.Dev = {sig_qdd}\n\n')#print(test_qd)
        result_qd[sn+1, :] = assble_qd[num_array[0], :]
        result_qdd[sn+1, :] = assble_qdd[num_array[0], :]
        energy_total = energy_total + energy_val[num_array[0]]['Energy']
        print(f'Energy consumption converges at {assble_qdd[num_array[0], :]} 1/s^2, total energy consumption: {energy_val[num_array[0]]} J\n')

        if sn == 0: # store optimized q, qd, qdd data at every time increment for plotting
            discrete_q = q_sample[num_array[0]]
            discrete_qd = qd_sample[num_array[0]]
            discrete_qdd = qdd_sample[num_array[0]]
        else:  # store optimized q, qd, qdd data at every time increment for plotting
            discrete_q = np.hstack((discrete_q, q_sample[num_array[0]]))
            discrete_qd = np.hstack((discrete_qd, qd_sample[num_array[0]]))
            discrete_qdd = np.hstack((discrete_qdd, qdd_sample[num_array[0]]))
    
    for joint_num in range(6): # write end angle, velocity and acceleration in result matrices
        result_q[-1, joint_num] = joint[joint_num].q[-1]
        result_qd[-1, joint_num] = joint[joint_num].qd[-1]
        result_qdd[-1, joint_num] = joint[joint_num].qdd[-1]

    time_vec[-1] = 3

    # In the following, the trajectory and energy consumption between the last two control points (from last sample point to the end of trajectory) are calculated and saved for plotting
    t_array = time_vec[-2:] # last two control points
    t_eval = np.linspace(t_array[0], t_array[1], width) # time array for evaluation, width = 10
    t_ref.append(t_eval) # store time array for plotting, this is different from the original trajectory time array because new time arrays are generated for each spline interpolation
    t_ref = np.concatenate(t_ref) # convert list to array for plotting
    print(t_trace)
    q_last = np.zeros((6, width)) # q-array between last two control points
    qd_last = np.zeros((6, width))
    qdd_last = np.zeros((6, width))

    for joint_num in range(6): # one more interpolation between last two control points, no HKA is needed because target positions are predetermined (i.e. robot should always reach the original target)
        if flag_stationary[joint_num] == True:
            q_last[joint_num, :] = q0[joint_num]
            qd_last[joint_num, :] = 0
            qdd_last[joint_num, :] = 0
        else:
            y1 = np.array([result_q[-2, joint_num], result_qd[-2, joint_num], result_qdd[-2, joint_num]]) # the angle, velocity and acceleration of previous point
            y2 = np.array([result_q[-1, joint_num], result_qd[-1, joint_num], result_qdd[-1, joint_num]]) # angle, velocity, acceleration of current point
            yi = np.vstack((y1, y2)) # vertical stack for interpolation with Berstein Polynomials
            q_bpoly = BPoly.from_derivatives(t_array, yi) # interpolation between previous and current point
            q_last[joint_num, :] = q_bpoly(t_eval) # angular trajectory
            qd_last[joint_num, :] = q_bpoly.derivative()(t_eval) # velocity trajectory
            qdd_last[joint_num, :] = q_bpoly.derivative(2)(t_eval) # acceleration trajectory      

    discrete_q = np.hstack((discrete_q, q_last)) # now we complete the entire optimized trajectory
    discrete_qd = np.hstack((discrete_qd, qd_last))
    discrete_qdd = np.hstack((discrete_qdd, qdd_last))

    q_torque_calc = np.transpose(q_last) # transpose for torque calculation
    qd_torque_calc = np.transpose(qd_last)
    qdd_torque_calc = np.transpose(qdd_last)
    power_val = [] # new initialization of power output list
    for k in range(width):
        torq_vec = cal_tau(q_torque_calc[k, :], qd_torque_calc[k, :], qdd_torque_calc[k, :]) # calculate torque
        velocity_vec = qd_torque_calc[k, :] # read velocity
        power_val.append(abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1))) # element-wise multiplication, then take 1-Norm to get robot power output

    energy_total_intv = simpson(power_val, t_eval) # integrate power over time to get energy using Simpson method
    energy_total = energy_total + energy_total_intv # the final total energy consumption over the entire trajectory

    # Below the original energy consumption is calculated based on the original trajectory
    time_interval_last = time[ctr:] # time array between last two control points based on time array of the original traj
    q_interval_last = np.zeros((6, len(time_interval_last))) 
    qd_interval_last = np.zeros((6, len(time_interval_last)))
    qdd_interval_last = np.zeros((6, len(time_interval_last)))

    for joint_num in range(6): # slice the last section (between last two control points) off of the original trajectory
        q_interval_last[joint_num, :] = joint[joint_num].q[ctr:]
        qd_interval_last[joint_num, :] = joint[joint_num].qd[ctr:]
        qdd_interval_last[joint_num, :] = joint[joint_num].qdd[ctr:]

    q_torque_calc = np.transpose(q_interval_last) # transpose for torque calculation
    qd_torque_calc = np.transpose(qd_interval_last)
    qdd_torque_calc = np.transpose(qdd_interval_last)

    power_val = [] # new initialization of power output again
    for k in range(len(time_interval_last)):
        torq_vec = cal_tau(q_torque_calc[k, :], qd_torque_calc[k, :], qdd_torque_calc[k, :]) # calculate torque
        velocity_vec = qd_torque_calc[k, :] # read velocity
        power_val.append(abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1))) # element-wise multiplication, then take 1-Norm to get robot power output
    energy_og_last = simpson(power_val, time_interval_last)
    energy_og_total = energy_og_total + energy_og_last # the total energy consumption over the entire original trajectory

    print(f'Optimization ended.\nOriginal energy consumption of the given trajectory is: {energy_og_total} J.\nTotal energy consumption of the optimizied trajectory is: {energy_total} J.\n')
    np.savetxt("result_q.txt", result_q)
    np.savetxt("result_qd.txt", result_qd)
    np.savetxt("result_qdd.txt", result_qdd)
    np.savetxt("time_vec.txt", time_vec)
    np.savetxt("t_int_trace.txt", t_ref)
    np.savetxt("nopenalty_q.txt", discrete_q)
    np.savetxt("nopenalty_qd.txt", discrete_qd)
    np.savetxt("nopenalty_qdd.txt", discrete_qdd)
    np.savetxt("discrete_time_vec.txt", time_vec)
    end_time = ti.time()
    print(end_time - start_time)
    print(iter_total)



    #print(t_ref)
    #print(f'Optimization ended.\nOriginal energy consumption of the given trajectory is: {energy_og_total} J.\nTotal energy consumption of the optimizied trajectory is: {energy_total} J.\n')

    #np.savetxt("result_q_int.txt", result_q)
    #np.savetxt("result_qd_int.txt", result_qd)
    #np.savetxt("result_qdd_int.txt", result_qdd)
    #np.savetxt("time_vec_int.txt", time_vec)    

    return result_q, result_qd, result_qdd, time_vec, joint

start1 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end1 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start2 = np.array([pi/2, -pi/2, pi/2, -pi/2, -pi/2, 0])
end2 = np.array([pi, -pi, 0, pi/4, -pi/2, pi])

start3 = np.array([0, 0, 0, 0, 0, 0])
end3 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])

start4 = np.array([pi, -pi/2, pi/2, -pi/2, -pi/2, 0])
end4 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start5 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end5 = np.array([2*pi/3, -pi/8, pi, -pi/2, 0, -pi/3])

#start = np.array([-pi, -pi, pi/2, -pi/2, -pi/2, 0])
#end = np.array([0, -0.749*pi, 0.69*pi, 0.444*pi, -0.8*pi, -pi])
joint = generate_traj_time(2, 201, start1, end1)
results = heuristic_kalman(40, 4, 6, 6, joint)
"""
joint = results[4]
time = results[3][0:7]
q = results[0]
qd = results[1]
qdd = results[2]
width = 10
qdd_graph = np.zeros((6, width))
qd_graph = np.zeros((6, width))
q_graph = np.zeros((6, width))
#print(time)





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
"""