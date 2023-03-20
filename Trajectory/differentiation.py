import roboticstoolbox as rtb
import numpy as np
import math
from math import pi
from roboticstoolbox import tools as tools
import matplotlib.pyplot as plt
from numpy import matlib
from call_tau import *
from traj import *
from tolerance import *
from lerp import *
from scipy.stats import truncnorm
from scipy.interpolate import BPoly
from scipy.integrate import simpson

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

def heuristic_kalman(N, Nbest, D, alpha, sd, n, sample_num, traj_time):
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
    sig_q = np.zeros(n)
    mu_qd = np.zeros(n)
    sig_qd = np.zeros(n)
    mu_qdd = np.zeros(n)
    sig_qdd = np.zeros(n)
    result = np.zeros(n+1)
    max_iter = 150
    lb = matlib.repmat(D[0, :], N, 1)
    ub = matlib.repmat(D[1, :], N, 1)
    asm_q = np.zeros((N, n))
    asm_qd = np.zeros((N, n))
    asm_qdd = np.zeros((N, n))
    q_trace = np.zeros((sample_num, 6, 50))
    qd_trace = np.zeros((6, 6, 50))
    qdd_trace = np.zeros((6, 6, 50))
    t_trace = []
    asm_qdd_temp = np.zeros((N, n))
    asm_bpoly = np.zeros((N, 3, 6))
    result_q = np.zeros((sample_num+2, 6))
    result_qd = np.zeros((sample_num+2, 6))
    result_qdd = np.zeros((sample_num+2, 6))
    array_q = np.zeros(6)
    array_qd = np.zeros(6)
    array_qdd = np.zeros(6)
    end_q = np.zeros(6)
    end_qd = np.zeros(6)
    end_qdd = np.zeros(6)
    debug = 0
    joint = generate_traj_time(traj_time) # generate trapezoidal trajectory with given trajectory time, discretized with 100 time steps, change config in traj.py
    u = np.zeros(6) # profile identifier, 1 for positive trapeze, -1 for negative trapeze
    time = joint[6]
    tolerance_band = np.zeros((2, 100*traj_time+1, 6))
    upper_bound = np.zeros(6)
    lower_bound = np.zeros(6)
    flag = True

    for i in range(6): # generate tolerance band for all joints
        angle = joint[i].q
        velocity = joint[i].qd
        upper = create_tolerance_bands(angle, velocity, time, 0.9, "upper")
        #print(upper[1])
        lower = create_tolerance_bands(angle, velocity, time, 0.9, "lower")
        tolerance_band[0, :, i] = upper[1]
        tolerance_band[1, :, i] = lower[1]

    t_accel = lower[2]
    t_brake = lower[3]
    time_vec = np.round(np.array([0, t_accel/2, t_accel-0.01, t_accel+(t_brake-t_accel)/3, t_accel+2*(t_brake-t_accel)/3, t_brake+0.01, t_brake+t_accel/2]), 2)
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

    for sn in range(6):

        energy_og = 0 # initialize pre-optimization energy consumption
        iter = 0 # initialize iteration counter  
        t1 = time_vec[sn]
        t2 = time_vec[sn+1]
        t_array = np.array([t1, t2])
        q_ref = np.zeros(6)
        ref_index = np.where(np.round(joint[6], 2) == np.round(t2, 2))  

        for ii in range(6): # fill out q_ref with the respective (original) joint angles
            q_ref[ii] = joint[ii].q[ref_index[0]]

        while ctr < ref_index[0]: # iterate from current counter to mu_index (the index that corresponds to the current sample point), counter is NOT reset after iteration is done

            for joint_num in range(6): # generate q, qd, qdd arrays for torque calculation with cal_tau()

                array_q[joint_num] = joint[joint_num].q[ctr]
                array_qd[joint_num] = joint[joint_num].qd[ctr]
                array_qdd[joint_num] = joint[joint_num].qdd[ctr]
            
            torq_og = cal_tau(array_q, array_qd, array_qdd)
            power_og = abs(np.multiply(torq_og, array_qd))
            energy_og = energy_og + np.linalg.norm(power_og, 1) * dt
            ctr = ctr + 1

        print(f'Calculation stops at trajectory time {joint[6][ctr]} s\n')
        print(f'Original energy consumption until sample point {sn+1} is: {energy_og} J\n')

        #if sn == sample_num - 1:
        #    energy_og = 0
        energy_og_total = energy_og_total + energy_og

        for i in range(n): # initialize mean, std.dev., and upper & lower bound for all joints

            mu_q[i] = joint[i].q[ref_index]
            sig_q[i] = 2.1
            mu_qd[i] = joint[i].qd[ref_index]
            sig_qd[i] = 5
            mu_qdd[i] = joint[i].qdd[ref_index]
            sig_qdd[i] = 2.6
            upper_bound[i] = tolerance_band[0, ref_index, i]
            lower_bound[i] = tolerance_band[1, ref_index, i]
            
        print(f'Debugging: sigma through initialisation: {sig_qdd[i]}\n') 
        #print(f'Original joint angle vector:\n{mu_q}\n')
        print(f'Original joint acceleration vector:\n{mu_qdd}\n')
        print(f'Original joint angle vector:\n{mu_q}\n')
        print(f'Lower bound = {lower_bound}\nUpper bound = {upper_bound}\n')

        while iter <= 150: # begin kalman iteration

            width = 50
            qdd_sample = np.zeros((N, 6, width))
            qd_sample = np.zeros((N, 6, width))
            q_sample = np.zeros((N, 6, width))
            t_eval = np.linspace(t1, t2, width)
            delta_t = t_eval[1] - t_eval[0]
            energy_val_def = [('Energy','f8'), ('Number','i2')] # list that contains the energy consumption and the corresponding row index from assble_qdd
            energy_val = np.zeros((N), dtype = energy_val_def)

            for joint_num in range(6): # generate random q, qd, qdd matrix for all joints
                flag = True
                #if joint[joint_num].q[0] > joint[joint_num].q[-1]:
                #    flag = False
                #if sn < 2 and flag == True:
                #    clip_left = (joint[joint_num].q[ref_index] + upper_bound[joint_num]) / 2 # considering the tolerance band for joint angle
                #    clip_right = upper_bound[joint_num] # considering the tolerance band for joint angle
                #elif sn < 2 and flag == False:
                #    clip_left = lower_bound[joint_num]
                #    clip_right = joint[joint_num].q[ref_index]
                #else:
                clip_left = lower_bound[joint_num] # considering the tolerance band for joint angle
                clip_right = upper_bound[joint_num] # considering the tolerance band for joint angle

                lb = (clip_left - mu_q[joint_num]) / sig_q[joint_num] # lower bound of the truncated gaussian distribution
                ub = (clip_right - mu_q[joint_num]) / sig_q[joint_num] # upper bound of the truncated gaussian distribution
                q_gauss = truncnorm(lb, ub, loc=mu_q[joint_num], scale=sig_q[joint_num]) # truncated gaussian distribution
                asm_q[:, joint_num] = q_gauss.rvs(size=N) # truncated gaussian distribution

                left_d = 0.8 * joint[joint_num].qd[ref_index]
                right_d = 1.2 * joint[joint_num].qd[ref_index]
                if left_d < right_d:
                    clip_left_d = left_d
                    clip_right_d = right_d
                else:
                    clip_left_d = right_d
                    clip_right_d = left_d

                lb_d = (clip_left_d - mu_qd[joint_num]) / sig_qd[joint_num] # lower bound of the truncated gaussian distribution
                ub_d = (clip_right_d - mu_qd[joint_num]) / sig_qd[joint_num] # upper bound of the truncated gaussian distribution
                qd_gauss = truncnorm(lb_d, ub_d, loc=mu_qd[joint_num], scale=sig_qd[joint_num])
                asm_qd[:, joint_num] = qd_gauss.rvs(size=N) # standard gaussian distribution for joint velocity and acceleration

                if sn < 2:

                    left_dd = 0.5 * joint[joint_num].qdd[ref_index]
                    right_dd = np.inf

                    if joint[joint_num].qdd[ref_index] == 0:
                        asm_qdd[:, joint_num] = np.random.normal(mu_qdd[joint_num], sig_qdd[joint_num], size=N)
                    elif left_dd < 0:
                        clip_left_dd = -np.inf
                        clip_right_dd = left_dd    
                    else:             
                        clip_left_dd = left_dd
                        clip_right_dd = right_dd

                    lb_dd = (clip_left_dd - mu_qdd[joint_num]) / sig_qdd[joint_num] # lower bound of the truncated gaussian distribution
                    ub_dd = (clip_right_dd - mu_qdd[joint_num]) / sig_qdd[joint_num] # upper bound of the truncated gaussian distribution
                    qdd_gauss = truncnorm(lb_dd, ub_dd, loc=mu_qdd[joint_num], scale=sig_qdd[joint_num])
                    asm_qdd[:, joint_num] = qdd_gauss.rvs(size=N)
                else:
                    asm_qdd[:, joint_num] = np.random.normal(mu_qdd[joint_num], sig_qdd[joint_num], size=N)
            
            #print(asm_qd)
            #print(asm_qdd)

            for i in range(N):

                energy_total_intv = 0
                power_val = []
                for joint_num in range(6):
                    y1 = np.array([result_q[sn, joint_num], result_qd[sn, joint_num], result_qdd[sn, joint_num]])
                    y2 = np.array([asm_q[i, joint_num], asm_qd[i, joint_num], asm_qdd[i, joint_num]])
                    yi = np.vstack((y1, y2))
                    q_bpoly = BPoly.from_derivatives(t_array, yi)
                    q_sample[i, joint_num, :] = q_bpoly(t_eval)
                    qd_sample[i, joint_num, :] = q_bpoly.derivative()(t_eval)
                    qdd_sample[i, joint_num, :] = q_bpoly.derivative(2)(t_eval)

                q_torque_calc = np.transpose(q_sample[i, :, :])
                qd_torque_calc = np.transpose(qd_sample[i, :, :])
                qdd_torque_calc = np.transpose(qdd_sample[i, :, :])
                #if i == 45:
                #    print(q_torque_calc)

                for k in range(width):
                    torq_vec = cal_tau(q_torque_calc[k, :], qd_torque_calc[k, :], qdd_torque_calc[k, :])
                    velocity_vec = qd_torque_calc[k, :]
                    power_val.append(abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1))) # element-wise multiplication, then take 1-Norm
                    #energy_total_intv = energy_total_intv + power_val * delta_t

                energy_total_intv = simpson(power_val, t_eval)
                energy_val[i] = (energy_total_intv, i)       

            sorted_energy_val = np.sort(energy_val, order='Energy')
            # sorted_cost_total = np.sort(cost_total_intv, order = 'Regulated Energy')
            #print(f'comparison: sorted cost \n{sorted_cost_total}')
            #print(f'comparison: sorted energy \n{sorted_energy_val}')
            #print(sorted_energy_val)
            post_q = np.zeros((Nbest, 6))
            post_qd = np.zeros((Nbest, 6))
            post_qdd = np.zeros((Nbest, 6))
            num_array = sorted_energy_val['Number']
            for i in range(Nbest):
                num = num_array[i] # returns the index for num-th best candidate
                post_q[i, :] = asm_q[num, :] # place the num th row of the assembled q-matrix onto the i-th row of the post_q matrix
                post_qd[i, :] = asm_qd[num, :]
                post_qdd[i, :] = asm_qdd[num, :]

            # measurement step
            mu_q_meas = np.mean(post_q, 0)
            mu_qd_meas = np.mean(post_qd, 0)
            mu_qdd_meas = np.mean(post_qdd, 0)
            var_q_post = np.var(post_q, 0)
            var_qd_post = np.var(post_qd, 0)
            var_qdd_post = np.var(post_qdd, 0)

            #Compute Kalman gain for q, qd, qdd
            new_mu_sig_q = kalman_gain(sig_q, var_q_post, mu_q, mu_q_meas)
            mu_q = new_mu_sig_q[0]
            #print(mu_q)
            sig_q = new_mu_sig_q[1]

            new_mu_sig_qd = kalman_gain(sig_qd, var_qd_post, mu_qd, mu_qd_meas)
            mu_qd = new_mu_sig_qd[0]
            sig_qd = new_mu_sig_qd[1]

            new_mu_sig_qdd = kalman_gain(sig_qdd, var_qdd_post, mu_qdd, mu_qdd_meas)
            mu_qdd = new_mu_sig_qdd[0]
            sig_qdd = new_mu_sig_qdd[1]      
            if all(i < 1e-6 for i in var_qdd_post) == True:
                print(f'exited loop at iter = {iter}')
                break
            iter = iter + 1      

        result_q[sn+1, :] = asm_q[num_array[0], :]
        result_qd[sn+1, :] = asm_qd[num_array[0], :]
        result_qdd[sn+1, :] = asm_qdd[num_array[0], :]
        print(result_qdd)
        energy_total = energy_total + energy_val[num_array[0]]['Energy']
        print(f'post acceleration matrix post_qdd = \n{post_qdd}\nwith a variance of {var_qdd_post}\nmean = {mu_qdd_meas}\n\n')

        for joint_num in range(6):
            q_trace[sn, joint_num, :] = q_sample[num_array[0], joint_num, :]
            qd_trace[sn, joint_num, :] = qd_sample[num_array[0], joint_num, :]
            qdd_trace[sn, joint_num, :] = qdd_sample[num_array[0], joint_num, :]

        t_trace.append(t_eval)
        print(f'Energy consumption converges at {asm_qdd[num_array[0], :]} 1/s^2\n\nJoint config: {asm_q[num_array[0], :]} total energy consumption: {energy_val[num_array[0]]} J\n')


    #print(result_q)
    #print(result_qd)
    #print(result_qdd)
    #print(t_ref)
    q_sample_ls = np.zeros((6, width))
    qd_sample_ls = np.zeros((6, width))
    qdd_sample_ls = np.zeros((6, width))
    #print(q_sample)
    for joint_num in range(6):
        result_q[-1, joint_num] = joint[joint_num].q[-1]
        result_qd[-1, joint_num] = joint[joint_num].qd[-1]
        result_qdd[-1, joint_num] = joint[joint_num].qdd[-1]
    """
    t_array_last = time_vec[-2:]
    t_eval_last = np.linspace(time_vec[-2], time_vec[-1], num=width)
    for joint_num in range(6):
        y1 = np.array([result_q[-2, joint_num], result_qd[-2, joint_num], result_qdd[-2, joint_num]])
        y2 = np.array([result_q[-1, joint_num], result_qd[-1, joint_num], result_qdd[-1, joint_num]])
        yi = np.vstack((y1, y2))
        q_bpoly_last = BPoly.from_derivatives(t_array_last, yi)
        q_sample_ls[joint_num, :] = q_bpoly_last(t_eval_last)
        qd_sample_ls[joint_num, :] = q_bpoly_last.derivative()(t_eval_last)
        qdd_sample_ls[joint_num, :] = q_bpoly_last.derivative(2)(t_eval)

    q_torque_calc_ls = np.transpose(q_sample_ls)
    qd_torque_calc_ls = np.transpose(qd_sample_ls)
    qdd_torque_calc_ls = np.transpose(qdd_sample_ls)
    #if i == 45:
    #    print(q_torque_calc)
    energy_ls = 0
    for k in range(width):
        torq_vec = cal_tau(q_torque_calc_ls[k, :], qd_torque_calc_ls[k, :], qdd_torque_calc_ls[k, :])
        velocity_vec = qd_torque_calc[k, :]
        power_val = abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1)) # element-wise multiplication, then take 1-Norm
        energy_ls = energy_ls + power_val * delta_t

    energy_total = energy_total + energy_ls
    energy_og_ls = 0
    while ctr < len(joint[6])-1: # iterate from current counter to mu_index (the index that corresponds to the current sample point), counter is NOT reset after iteration is done

        for joint_num in range(6): # generate q, qd, qdd arrays for torque calculation with cal_tau()

            array_q[joint_num] = joint[joint_num].q[ctr]
            array_qd[joint_num] = joint[joint_num].qd[ctr]
            array_qdd[joint_num] = joint[joint_num].qdd[ctr]
        
        torq_og_ls = cal_tau(array_q, array_qd, array_qdd)
        power_og_ls = abs(np.multiply(torq_og_ls, array_qd))
        energy_og_ls = energy_og_ls + np.linalg.norm(power_og_ls, 1) * dt
        ctr = ctr + 1    

    energy_og_total = energy_og_total + energy_og_ls    
    """
    time_vec = np.append(time_vec, traj_time)
    print(f'Optimization ended.\nOriginal energy consumption of the given trajectory is: {energy_og_total} J.\nTotal energy consumption of the optimizied trajectory is: {energy_total} J.\n')
    print(result_qdd)

    np.savetxt("result_q.txt", result_q)
    np.savetxt("result_qd.txt", result_qd)
    np.savetxt("result_qdd.txt", result_qdd)
    np.savetxt("time_vec.txt", time_vec)

    return result_q, result_qd, result_qdd, time_vec, joint, q_trace, t_trace, qd_trace, qdd_trace


results = heuristic_kalman(50, 5, np.array([[0, 0, 0, 0, 0, 0], [0.0003, 5, 6, 8, 20, 10]]), 0, 0, 6, 6, 2)
"""
result_q = results[0]
result_qd = results[1]
result_qdd = results[2]

xi = results[3]


#print(xi)
for joint_num in range(6):
    y = result_q[:, joint_num]
    yd = result_qd[:, joint_num]
    ydd = result_qdd[:, joint_num]
    yi = np.vstack((y, yd, ydd))
    yi = np.transpose(yi)
    #print(yi)

    func = BPoly.from_derivatives(xi, yi)
    t_evaluate = np.linspace(xi[0], xi[-1], num=200)
    q_p[joint_num, :] = func(t_evaluate)
    qd_p[joint_num, :] = func.derivative()(t_evaluate)
    qdd_p[joint_num, :] = func.derivative(2)(t_evaluate)

    time = np.array(results[6])
    q_org = results[4][0].q
    qd_org = results[4][0].qd
    qdd_org = results[4][0].qdd
    time1 = results[4][6]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')

ax1.plot(time1, q_org)
ax1.plot(t_evaluate, q_p)

ax2.plot(time1, qd_org)
ax2.plot(t_evaluate, qd_p)

ax3.plot(time1, qdd_org)
ax3.plot(t_evaluate, qdd_p)
plt.show()

"""


# Below is a whole section of image output....
"""
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(time, joint[0].q, label='original joint1', color='red')
ax1.plot(upper[0], tolerance_band[0, :, 0], label='tolerance band', linestyle='dashed', color='green')
ax1.plot(upper[0], tolerance_band[1, :, 0], linestyle='dashed', color='green')
ax1.legend()
ax2.plot(time, joint[1].q, label='original joint2', color='red')
ax2.plot(upper[0], tolerance_band[0, :, 1], label='tolerance band', linestyle='dashed', color='green')
ax2.plot(upper[0], tolerance_band[1, :, 1], linestyle='dashed', color='green')
ax2.legend()
ax3.plot(time, joint[2].q, label='original joint3', color='red')
ax3.plot(upper[0], tolerance_band[0, :, 2], label='tolerance band', linestyle='dashed', color='green')
ax3.plot(upper[0], tolerance_band[1, :, 2], linestyle='dashed', color='green')
ax3.legend()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(time, joint[3].q, label='original joint1', color='red')
ax1.plot(upper[0], tolerance_band[0, :, 3], label='tolerance band', linestyle='dashed', color='green')
ax1.plot(upper[0], tolerance_band[1, :, 3], linestyle='dashed', color='green')
ax1.legend()
ax2.plot(time, joint[4].q, label='original joint2', color='red')
ax2.plot(upper[0], tolerance_band[0, :, 4], label='tolerance band', linestyle='dashed', color='green')
ax2.plot(upper[0], tolerance_band[1, :, 4], linestyle='dashed', color='green')
ax2.legend()
ax3.plot(time, joint[5].q, label='original joint3', color='red')
ax3.plot(upper[0], tolerance_band[0, :, 5], label='tolerance band', linestyle='dashed', color='green')
ax3.plot(upper[0], tolerance_band[1, :, 5], linestyle='dashed', color='green')
ax3.legend()

plt.show()
"""