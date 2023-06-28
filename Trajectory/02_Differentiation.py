'''
@author: Shihui Liu

Use tolerance_band_diff() to generate tolerance band around the original angular trajectory and optimize energy consumption using heuristic Kalman and differential dynamic programming.
Returns optimized trajectories and optimized energy consumption.
Results are additionally saved to result_q.txt, result_qd.txt and result_qdd.txt in root directory.
'''

import numpy as np
from math import pi
from roboticstoolbox import tools as tools
from call_tau import *
from traj import *
from tolerance import *
from lerp import *
from scipy.stats import truncnorm
from scipy.interpolate import BPoly
from scipy.integrate import simpson
from HKA_kalman_gain import *
import time as ti

def tolerance_band_diff(N, Nbest, n, sample_num, traj):
    """
    Applies heuristic Kalman algorithm to a given trajectory of n robot joints,
    returns the optimized q, qd, qdd trajectories for all joints

    :param N: number of random values to be generated
    :param Nbest: number of best candidates to consider
    :param n: number of joints
    :sample_num: number of sample points on the trajectory to be evaluated, excluding start and end point
    :traj: original trajectory
    
    :return:
    :result_q, result_qd, result_qdd: optimized joint angle, velocity and acceleration, (sample_num+2) x 6 matrix
    :time_vec: time vector for creating graphs, 1 x (sample_num + 2) array, ranging from 0 sec to traj_steps sec
    """
    flag = np.full(6, False)
    mu_q = np.zeros(n)
    sig_q = np.zeros(n)
    mu_qd = np.zeros(n)
    sig_qd = np.zeros(n)
    mu_qdd = np.zeros(n)
    sig_qdd = np.zeros(n)
    asm_q = np.zeros((N, n))
    asm_qd = np.zeros((N, n))
    asm_qdd = np.zeros((N, n))
    t_trace = []
    result_q = np.zeros((sample_num+2, 6))
    result_qd = np.zeros((sample_num+2, 6))
    result_qdd = np.zeros((sample_num+2, 6))
    array_q = np.zeros(6)
    array_qd = np.zeros(6)
    array_qdd = np.zeros(6)
    time = traj[6]
    q0 = np.zeros(6)
    qf = np.zeros(6)
    width = 50 # number of increments between two control points for energy calculation. width = 50 delivers similar accuracy to numerical integration methods like Simpson
    tolerance_band = np.zeros((2, 201, 6))
    upper_bound = np.zeros(6)
    lower_bound = np.zeros(6)
    discrete_q = np.zeros((6, width))
    discrete_qd = np.zeros((6, width))
    discrete_qdd = np.zeros((6, width))
    iter_total = 0
    start_time = ti.time()

    for i in range(6): # generate tolerance band for all joints
        angle = traj[i].q
        velocity = traj[i].qd
        q0[i] = angle[0]
        qf[i] = angle[-1]
        if q0[i] == qf[i]:
            flag[i] = True

        else:
            upper = create_tolerance_bands(angle, velocity, time, 0.7, "upper") # see tolerance.py
            lower = create_tolerance_bands(angle, velocity, time, 0.7, "lower")
            tolerance_band[0, :, i] = upper[1]
            tolerance_band[1, :, i] = lower[1]

    t_accel = lower[2]
    t_brake = lower[3]
    time_vec = np.round(np.array([0, t_accel/2, t_accel-0.01, t_accel+(t_brake-t_accel)/3, t_accel+2*(t_brake-t_accel)/3, t_brake+0.01, t_brake+t_accel/2]), 2) # manually define control points based on acceleration and brake time
    ctr = 0
    dt = traj[6][1] - traj[6][0]
    energy_og_total = 0 # accumulated energy consumption along unoptimized trajectory
    energy_total = 0 # accumulated energy consumption along optimized trajectory

    for i in range(n): # write initial angle, velocity and acceleration for all joints in the first rows of the result lists
        result_q[0, i] = traj[i].q[0]
        result_qd[0, i] = traj[i].qd[0]
        result_qdd[0, i] = traj[i].qdd[0]

    for sn in range(6): # iterate through all control points

        energy_og = 0 # initialize pre-optimization energy consumption
        iter = 0 # initialize iteration counter  
        t1 = time_vec[sn]
        t2 = time_vec[sn+1]
        t_array = np.array([t1, t2]) # base x points for Bernstein polynomial interpolation
        q_ref = np.zeros(6)
        ref_index = np.where(np.round(traj[6], 2) == np.round(t2, 2)) # find control point in time array

        for ii in range(6): # fill out q_ref with the respective (original) joint angles
            q_ref[ii] = traj[ii].q[ref_index[0]] # joint angle at control point

        while ctr < ref_index[0]: # iterate from current counter to mu_index (the index that corresponds to the current sample point), counter must NOT be reset after iteration is done
            for joint_num in range(6): # generate q, qd, qdd arrays for torque calculation with cal_tau()
                array_q[joint_num] = traj[joint_num].q[ctr]
                array_qd[joint_num] = traj[joint_num].qd[ctr]
                array_qdd[joint_num] = traj[joint_num].qdd[ctr]
            
            torq_og = cal_tau(array_q, array_qd, array_qdd) # torque calculation, see call_tau.py
            power_og = abs(np.multiply(torq_og, array_qd)) # element-wise multiplication 
            energy_og = energy_og + np.linalg.norm(power_og, 1) * dt # energy consumption in time increment dt
            ctr = ctr + 1

        print(f'Calculation stops at trajectory time {traj[6][ctr]} s\n')
        print(f'Original energy consumption until sample point {sn+1} is: {energy_og} J\n')
        energy_og_total = energy_og_total + energy_og

        for i in range(n): # initialize mean, std.dev., and upper & lower bound for all joints
            mu_q[i] = traj[i].q[ref_index]
            sig_q[i] = 2.1
            mu_qd[i] = traj[i].qd[ref_index]
            sig_qd[i] = 5
            mu_qdd[i] = traj[i].qdd[ref_index]
            sig_qdd[i] = 2.6
            upper_bound[i] = tolerance_band[0, ref_index, i]
            lower_bound[i] = tolerance_band[1, ref_index, i]
            
        print(f'Debugging: sigma through initialisation: {sig_qdd[i]}\n') 
        print(f'Original joint acceleration vector:\n{mu_qdd}\n')
        print(f'Original joint angle vector:\n{mu_q}\n')
        print(f'Lower bound = {lower_bound}\nUpper bound = {upper_bound}\n')

        while iter <= 150: # begin kalman iteration
            qdd_sample = np.zeros((N, 6, width))
            qd_sample = np.zeros((N, 6, width))
            q_sample = np.zeros((N, 6, width))
            t_eval = np.linspace(t1, t2, width)
            energy_val_def = [('Energy','f8'), ('Number','i2')] # list that contains the energy consumption and the corresponding row index from assble_qdd
            energy_val = np.zeros((N), dtype = energy_val_def)

            for joint_num in range(6): # generate random q, qd, qdd matrix for all joints
                if flag[joint_num] == True:
                    pass
                else:
                    clip_left = lower_bound[joint_num] # considering the tolerance band for joint angle
                    clip_right = upper_bound[joint_num] # considering the tolerance band for joint angle
                    lb = (clip_left - mu_q[joint_num]) / sig_q[joint_num] # lower bound of the truncated gaussian distribution for angle
                    ub = (clip_right - mu_q[joint_num]) / sig_q[joint_num] # upper bound of the truncated gaussian distribution for angle
                    q_gauss = truncnorm(lb, ub, loc=mu_q[joint_num], scale=sig_q[joint_num]) # truncated gaussian distribution for angle
                    asm_q[:, joint_num] = q_gauss.rvs(size=N) # angle distribution for all joints
                    left_d = 0.8 * traj[joint_num].qd[ref_index] # empirically define the boundaries for joint velocity
                    right_d = 1.2 * traj[joint_num].qd[ref_index] # empirically define the boundaries for joint velocity

                    if left_d < right_d: # if velocity is positive
                        clip_left_d = left_d
                        clip_right_d = right_d
                    else: # velocity is negative, upper and lower bound need to be swapped
                        clip_left_d = right_d
                        clip_right_d = left_d

                    lb_d = (clip_left_d - mu_qd[joint_num]) / sig_qd[joint_num] 
                    ub_d = (clip_right_d - mu_qd[joint_num]) / sig_qd[joint_num] 
                    qd_gauss = truncnorm(lb_d, ub_d, loc=mu_qd[joint_num], scale=sig_qd[joint_num])
                    asm_qd[:, joint_num] = qd_gauss.rvs(size=N) 

                    if sn < 2: # acceleration phase
                        left_dd = 0.5 * traj[joint_num].qdd[ref_index] # acceleration cannot be lower than half of its original value
                        right_dd = np.inf # maximal acceleration not limited
                        if traj[joint_num].qdd[ref_index] == 0: # if joint is stationary 
                            asm_qdd[:, joint_num] = np.random.normal(mu_qdd[joint_num], sig_qdd[joint_num], size=N)
                        elif left_dd < 0: # if acceleration is negative, upper and lower bound need to be swapped
                            clip_left_dd = -np.inf
                            clip_right_dd = left_dd    
                        else: # acceleration is positive             
                            clip_left_dd = left_dd
                            clip_right_dd = right_dd

                        lb_dd = (clip_left_dd - mu_qdd[joint_num]) / sig_qdd[joint_num] # lower bound of the truncated gaussian distribution for acceleration
                        ub_dd = (clip_right_dd - mu_qdd[joint_num]) / sig_qdd[joint_num] # upper bound of the truncated gaussian distribution for acceleration
                        qdd_gauss = truncnorm(lb_dd, ub_dd, loc=mu_qdd[joint_num], scale=sig_qdd[joint_num])
                        asm_qdd[:, joint_num] = qdd_gauss.rvs(size=N)
                    else:
                        asm_qdd[:, joint_num] = np.random.normal(mu_qdd[joint_num], sig_qdd[joint_num], size=N)
            
            for i in range(N):
                energy_total_intv = 0
                power_val = []
                for joint_num in range(6):
                    if flag[joint_num] == True:
                        q_sample[i, joint_num, :] = q0[joint_num]
                        qd_sample[i, joint_num, :] = 0
                        qdd_sample[i, joint_num, :] = 0
                    else:
                        y1 = np.array([result_q[sn, joint_num], result_qd[sn, joint_num], result_qdd[sn, joint_num]]) # the angle, velocity and acceleration of previous point
                        y2 = np.array([asm_q[i, joint_num], asm_qd[i, joint_num], asm_qdd[i, joint_num]]) # angle, velocity, acceleration of current point
                        yi = np.vstack((y1, y2)) # vertical stack for BPoly operation
                        q_bpoly = BPoly.from_derivatives(t_array, yi) # interpolation between previous and current point
                        q_sample[i, joint_num, :] = q_bpoly(t_eval) # angular trajectory
                        qd_sample[i, joint_num, :] = q_bpoly.derivative()(t_eval) # velocity trajectory
                        qdd_sample[i, joint_num, :] = q_bpoly.derivative(2)(t_eval) # acceleration trajectory

                q_torque_calc = np.transpose(q_sample[i, :, :]) # transpose for torque calculation
                qd_torque_calc = np.transpose(qd_sample[i, :, :])
                qdd_torque_calc = np.transpose(qdd_sample[i, :, :])

                for k in range(width):
                    torq_vec = cal_tau(q_torque_calc[k, :], qd_torque_calc[k, :], qdd_torque_calc[k, :]) # calculate torque
                    velocity_vec = qd_torque_calc[k, :] # read velocity
                    power_val.append(abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1))) # element-wise multiplication, then take 1-Norm to get robot power output

                energy_total_intv = simpson(power_val, t_eval) # integrate power over time to get energy using Simpson method
                energy_val[i] = (energy_total_intv, i)       

            sorted_energy_val = np.sort(energy_val, order='Energy') # sort energy consumption from lowest to highest
            post_q = np.zeros((Nbest, 6))
            post_qd = np.zeros((Nbest, 6))
            post_qdd = np.zeros((Nbest, 6))
            num_array = sorted_energy_val['Number']

            for i in range(Nbest): # Consider the Nbest candidates
                num = num_array[i] # returns the index for num-th best candidate
                post_q[i, :] = asm_q[num, :] # place the num th row of the assembled q-matrix onto the i-th row of the post_q matrix
                post_qd[i, :] = asm_qd[num, :]
                post_qdd[i, :] = asm_qdd[num, :]

            # measurement step, compute mean and variance for q, qd and qdd
            mu_q_meas = np.mean(post_q, 0)
            mu_qd_meas = np.mean(post_qd, 0)
            mu_qdd_meas = np.mean(post_qdd, 0)
            var_q_post = np.var(post_q, 0)
            var_qd_post = np.var(post_qd, 0)
            var_qdd_post = np.var(post_qdd, 0)

            # Compute Kalman gain for q, qd, qdd
            new_mu_sig_q = kalman_gain(sig_q, var_q_post, mu_q, mu_q_meas)
            mu_q = new_mu_sig_q[0]
            sig_q = new_mu_sig_q[1]
            new_mu_sig_qd = kalman_gain(sig_qd, var_qd_post, mu_qd, mu_qd_meas)
            mu_qd = new_mu_sig_qd[0]
            sig_qd = new_mu_sig_qd[1]
            new_mu_sig_qdd = kalman_gain(sig_qdd, var_qdd_post, mu_qdd, mu_qdd_meas)
            mu_qdd = new_mu_sig_qdd[0]
            sig_qdd = new_mu_sig_qdd[1]      

            if all(i < 1e-4 for i in var_qdd_post) == True: # convergence criterion
                print(f'exited loop at iter = {iter}')
                break

            iter = iter + 1      
        iter_total = iter_total + iter
        
        result_q[sn+1, :] = asm_q[num_array[0], :] # the result calculated by HKA is written to the result matrix
        result_qd[sn+1, :] = asm_qd[num_array[0], :] # the result calculated by HKA is written to the result matrix
        result_qdd[sn+1, :] = asm_qdd[num_array[0], :] # the result calculated by HKA is written to the result matrix

        if sn == 0: # store optimized q, qd, qdd data at every time increment for plotting
            discrete_q = q_sample[num_array[0]]
            discrete_qd = qd_sample[num_array[0]]
            discrete_qdd = qdd_sample[num_array[0]]
        else:  # store optimized q, qd, qdd data at every time increment for plotting
            discrete_q = np.hstack((discrete_q, q_sample[num_array[0]]))
            discrete_qd = np.hstack((discrete_qd, qd_sample[num_array[0]]))
            discrete_qdd = np.hstack((discrete_qdd, qdd_sample[num_array[0]]))

        energy_total = energy_total + energy_val[num_array[0]]['Energy']
        print(f'post acceleration matrix post_qdd = \n{post_qdd}\nwith a variance of {var_qdd_post}\nmean = {mu_qdd_meas}\n\n')
        t_trace.append(t_eval) # store time array for plotting, this is different from the original trajectory time array because new time arrays are generated for each spline interpolation
        print(f'Energy consumption converges at {asm_qdd[num_array[0], :]} 1/s^2\n\nJoint config: {asm_q[num_array[0], :]} total energy consumption: {energy_val[num_array[0]]} J\n')

    for joint_num in range(6): # write end angle, velocity and acceleration in result matrices
        result_q[-1, joint_num] = traj[joint_num].q[-1]
        result_qd[-1, joint_num] = traj[joint_num].qd[-1]
        result_qdd[-1, joint_num] = traj[joint_num].qdd[-1]

    time_vec = np.append(time_vec, traj[6][-1]) # write trajectory end time to time vector

    # Below the trajectory and energy consumption between the last two control points (from last sample point to the end of trajectory) are calculated and saved for plotting
    t_array = time_vec[-2:] # last two control points
    t_eval = np.linspace(t_array[0], t_array[1], width) # time array for evaluation, width = 50
    t_trace.append(t_eval) # store time array for plotting, this is different from the original trajectory time array because new time arrays are generated for each spline interpolation
    t_trace = np.concatenate(t_trace) # convert list to array for plotting
    #print(t_trace)
    q_last = np.zeros((6, width)) # q-array between last two control points
    qd_last = np.zeros((6, width))
    qdd_last = np.zeros((6, width))

    for joint_num in range(6): # one more interpolation between last two control points, no HKA is needed because target positions are predetermined (i.e. robot should always reach the original target)
        if flag[joint_num] == True:
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
        q_interval_last[joint_num, :] = traj[joint_num].q[ctr:]
        qd_interval_last[joint_num, :] = traj[joint_num].qd[ctr:]
        qdd_interval_last[joint_num, :] = traj[joint_num].qdd[ctr:]

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
    np.savetxt("t_trace.txt", t_trace)
    np.savetxt("discrete_q.txt", discrete_q)
    np.savetxt("discrete_qd.txt", discrete_qd)
    np.savetxt("discrete_qdd.txt", discrete_qdd)
    end_time = ti.time()
    print(end_time - start_time)
    print(iter_total)


    return result_q, result_qd, result_qdd, time_vec, traj, discrete_q, discrete_qd, discrete_qdd, t_trace

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

traj = generate_traj_time(2, 201, start1, end1)
results = tolerance_band_diff(30, 3, 6, 6, traj)