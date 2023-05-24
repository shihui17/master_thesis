import roboticstoolbox as rtb
import numpy as np
import math
from math import pi
from roboticstoolbox import tools as tools
import matplotlib.pyplot as plt
from numpy import matlib
from call_tau import *
from traj import *
from scipy.interpolate import interp1d
from tolerance import *
from lerp import *
from scipy.stats import truncnorm
import CubicEquationSolver

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
    a = 0.7 * (s**2 / (s**2 + np.max(P))) # slowdown factor to slow down the convergence, coefficient (in this case 0.6) can vary (see HKA book)
    #print(P)
    #print(var_q_post)
    sig_new = sig + a * (np.sqrt(P) - sig) # set new std. deviation
    mu_new = mu_post # set new mean
    #print(f'new mean: {mu_new}')
    
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
    assble_q = np.zeros((N, n))
    assble_qd = np.zeros((N, n))
    assble_qdd = np.zeros((N, n))
    assble_qdd_temp = np.zeros((N, n))
    result_q = np.zeros((sample_num+1, 6))
    result_qd = np.zeros((sample_num+1, 6))
    result_qdd = np.zeros((sample_num+1, 6))
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

    for i in range(6): # generate tolerance band for all joints
        angle = joint[i].q
        velocity = joint[i].qd
        upper = create_tolerance_bands(angle, velocity, time, 0.95, "upper")
        #print(upper[1])
        lower = create_tolerance_bands(angle, velocity, time, 0.95, "lower")
        tolerance_band[0, :, i] = upper[1]
        tolerance_band[1, :, i] = lower[1]

    t_accel = lower[2]
    t_brake = lower[3]
    time_vec = np.round(np.array([0, t_accel/2, t_accel-0.05, t_accel+(t_brake-t_accel)/3, t_accel+2*(t_brake-t_accel)/3, t_brake+0.05, t_brake+t_accel/2]), 2)
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

    for sn, time_ind in enumerate(range(6)): # now iterate through all sample points

        flag = False
        energy_og = 0 # initialize pre-optimization energy consumption
        iter = 0 # initialize iteration counter
        #t2 = traj_time/(sample_num+1)*(sn+1) # current sample point
        #time_vec[sn+1] = t2 # current sample point
        t1 = time_vec[time_ind] # previous sample point, we are at sample point sn+1 right now, meaning previous sample point is sn
        t2 = time_vec[time_ind+1]
        q_ref = np.zeros(6) # initialize reference joint angle array, where original joint angles at sample point sn are stored
        ref_index = np.where(np.round(joint[6], 2) == np.round(t2, 2))  

        for ii in range(6): # fill out q_ref with the respective (original) joint angles
            q_ref[ii] = joint[ii].q[ref_index[0]]

        #print(f'wtf am i printing ????{time_vec}')
        #mu_index_pre = math.floor(200/(sample_num+1)*sn)
        #mu_index = math.floor(200/(sample_num+1)*(sn+1))
        #print(f'TEST!: {time_vec}')
        #print(f'Calculating optimized trajectory at sample point {sn}, corresponding to trajectory time {traj_steps/(sample_num+1)*(sn+1)}\n')    

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

        # initialize mean vector mu_q, mu_qd, mu_qdd, and std vector sig_qdd
        for i in range(n):
            # print(sig_qdd[i])
            # tolerance_band[0, :, i] = upper[1]
            # tolerance_band[1, :, i] = lower[1]
            upper_bound[i] = tolerance_band[0, ref_index, i]
            lower_bound[i] = tolerance_band[1, ref_index, i]
            #print(f'upper bound {upper_bound}')
            #print(f'lower bound {lower_bound}')
            #print(joint_num)
            #print(mu_qdd[joint_num], sig_qdd[joint_num])
            #print(f'upper = {upper_bound}')
            #print(f'lower = {lower_bound}')
            mu_q[i] = joint[i].q[ref_index]
            mu_qd[i] = joint[i].qd[ref_index]
            mu_qdd[i] = joint[i].qdd[ref_index]
            #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
            if mu_qdd[i] == 0:
                sig_qdd[i] = 1.33 # initialize std vector, hard coded for now
                flag = True
            else:
                sig_qdd[i] = 1.33 # abs(1*mu_qdd[i])
            
        print(f'Debugging: sigma through initialisation: {sig_qdd[i]}\n')

        test_qd = mu_qd

        #print(f'Original joint angle vector:\n{mu_q}\n')
        print(f'Original joint acceleration vector:\n{mu_qdd}\n')
        print(f'Original joint angle vector:\n{mu_q}\n')
        print(f'Lower bound = {lower_bound}\nUpper bound = {upper_bound}\n')

        #ref_torq = cal_tau(mu_q, mu_qd, mu_qdd) # original output torque for reference
        #ref_power = np.multiply(ref_torq, mu_qd) # calculate orginal power output of each joint
        #ref_total_power = np.linalg.norm(ref_power, 1) # summation to get total power output
        #print(f'Original total power output: {ref_total_power} W\n')

        while iter <= 50: # begin kalman iteration
            
            assble_qdd_temp = np.zeros((N, 6))
            width = 10
            qdd_sample = np.zeros((6, width))
            qd_sample = np.zeros((6, width))
            q_sample = np.zeros((6, width))
            #print(f'current sig: {sig_qdd}')
            for joint_num in range(6): # generate gaussian destribution for acceleration of each joint (N random values), results written in assble_qdd (an Nx6 matrix)
                if result_qdd[0, joint_num] > 0:
                    flag = True # True -> positive profile, angle increases
                else:
                    flag = False # False -> negative profile, angle decreases
                for i in range(N):

                    identifier = 0
                    while True:

                        identifier += 1
                        if N*2 < identifier < N*2+2:
                            print(f'Tracking (sn = {sn}): we are (stuck) at joint {joint_num} with joint angle {q2}, qdd2 {qdd2}, sig = {sig_qdd[joint_num]}, mu = {mu_qdd[joint_num]}')
                            print(f'lower bound = {lower_bound[joint_num]}, upper bound = {upper_bound[joint_num]}')
                            #print(f'params for numerical integration: result_q = {result_q[sn, joint_num]}, result_qd = {result_qd[sn, joint_num]}, result_qdd = {result_qdd[sn, joint_num]}\n')
                            print(f'iteration: {iter}')
                            #print(f'post acceleration matrix post_qdd = \n{post_qdd}\nwith a variance of {var_qdd_post}\nmean = {mu_qdd_meas}\n\n')
                            x2 = t2
                            x1 = t1
                            y2 = qdd2
                            y1 = result_qdd[sn, joint_num]
                            k = (y2 - y1)/(x2 - x1)
                            qd0 = result_qd[sn, joint_num]
                            q0 = result_q[sn, joint_num]
                            xx = 1/6 * x2**3 - 1/2 * x1 * x2**2 + 1/2 * x1**2 * x2 - 1/6 * x1**3
                            bb = 1/2 * y1 * x2**2 + qd0 * x2 - x1 * y1 * x2 - 0.5 * y1 * x1**2 - qd0 * x1 + x1 * y1 * x1 + q0
                            yy = (lower_bound[joint_num] + upper_bound[joint_num]) / 2
                            kk = (yy - bb) / xx
                            qdd2 = kk * (x2 - x1) + y1
                            print(f'corrected qdd2 {qdd2}')
                            print(f'bound set at {0.6*joint[joint_num].qdd[ref_index]}')
                            print(f'{t1}, {t2}, {y1}, {qdd2}, {t2}, {qd0}, {q0}')
                            qt = lerp_func_double_integral(t1, t2, result_qdd[sn, joint_num], qdd2, t2, result_qd[sn, joint_num], result_q[sn, joint_num])
                            #print(f'why is bound not working {0.6*mu_qdd[joint_num]}')
                            #mu_qdd[joint_num] = qdd2
                            #print(f'{qt} = {yy} ??')
                            """
                            if q2 > upper_bound[joint_num]:
                                x2 = t2
                                x1 = t2
                                y2 = qdd2
                                y1 = result_qdd[sn, joint_num]
                                k = (y2 - y1)/(x2 - x1)
                                qd0 = result_qd[sn, joint_num]
                                q0 = result_q[sn, joint_num]
                                a2 = qd0 - 0.5 * k * x1**2 + k * x1 * x1 - y1 * x1
                                C = q0 - (1/6 * k * x1**3 - 0.5 * k * x1**3 + 0.5 * k * y1 * x1**2 + a2 * x1)
                                CubicEquationSolver.solve(1/6*k, -0.5*k*x1+0.5*y1, a2, C)
                                lb = (-5 - mu_qdd[joint_num]) / sig_qdd[joint_num]
                                ub = 0 / sig_qdd[joint_num]
                            elif q2 < lower_bound[joint_num]:
                                lb = 0
                                ub = (5 - mu_qdd[joint_num]) / sig_qdd[joint_num]      
                            """                          
                            #print(f'sig = {sig_qdd[joint_num]}')
                            
                        
                        elif mu_qdd[joint_num] < 0:
                            lb = (-10 - 0) / sig_qdd[joint_num]
                            #lb = (max(1.5 * mu_qdd[joint_num], -1.5 * abs(result_qdd[0, joint_num])) - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            ub = (10 - 0) / sig_qdd[joint_num]
                            #lb = (-10 - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            #lb = (max(1.5 * mu_qdd[joint_num], -1.5 * abs(result_qdd[0, joint_num])) - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            #ub = (10 - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            #print(f'1.5 * mu_qdd[{i}] = {1.5*mu_qdd[i]}; the other is {-1.5 * abs(result_qdd[0, i])}\n')
                            #print(f'lb = {lb}')
                            #ub = (min(0.8 * mu_qdd[joint_num], -0.8 * abs(result_qdd[0, joint_num])) - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            #assble_qdd[:, i] = np.transpose(s_qdd_trunc.rvs(size=N))
                            a = 0
                            #print(a)
                        elif mu_qdd[joint_num] > 0:
                            lb = (-10 - 0) / sig_qdd[joint_num]
                            #lb = (max(1.5 * mu_qdd[joint_num], -1.5 * abs(result_qdd[0, joint_num])) - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            ub = (10 - 0) / sig_qdd[joint_num]                            
                            #lb = (-10 - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            #lb = (max(0.8 * mu_qdd[joint_num], 0.8 * abs(result_qdd[0, joint_num])) - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            #ub = (10 - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            #ub = (min(1.5 * mu_qdd[joint_num], 1.5 * abs(result_qdd[0, joint_num])) - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            a = 1
                            #print(a)
                        else:
                            #lb = (-10 - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            #ub = (10 - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            lb = (-10 - 0) / sig_qdd[joint_num]
                            #lb = (max(1.5 * mu_qdd[joint_num], -1.5 * abs(result_qdd[0, joint_num])) - mu_qdd[joint_num]) / sig_qdd[joint_num]
                            ub = (10 - 0) / sig_qdd[joint_num]
                        
                        
                        #lb = (-5 - mu_qdd[joint_num]) / sig_qdd[joint_num]
                        #ub = (5 - mu_qdd[joint_num]) / sig_qdd[joint_num]
                        #print(f'sig = {sig_qdd[joint_num]}')
                        s_qdd_trunc = truncnorm(lb, ub, loc=0, scale = sig_qdd[joint_num])                        
                        #s_qdd_trunc = truncnorm(lb, ub, loc=mu_qdd[joint_num], scale = sig_qdd[joint_num])
                        qdd2 = s_qdd_trunc.rvs(size=1)
                        
                        #qdd2 = np.random.normal(mu_qdd[joint_num], sig_qdd[joint_num])
                        #if sn == 1 and joint_num == 0:
                            #print(f'qdd bisher for joint {joint_num}:\n{result_qdd}\n\nqd bisher for joint {joint_num}:\n{result_qd}\n\nq bisher for joint {joint_num}:\n{result_q}\n\n')
                            #print(f'lower bound = {lower_bound}, upper bound = {upper_bound}\nat sample point {sn+1}: q bisher for joint {joint_num}:\n{result_q[:, joint_num]}\n\n')
                        q2 = lerp_func_double_integral(t1, t2, result_qdd[sn, joint_num], qdd2, t2, result_qd[sn, joint_num], result_q[sn, joint_num])
                        #if sn == 2:
                        if lower_bound[joint_num] <= q2 <= upper_bound[joint_num]:
                            assble_qdd_temp[i, joint_num] = qdd2
                            #print(assble_qdd_temp)
                            break
                            
                #if joint_num == 5:
                    #print(f'upper bound {upper_bound}')
                    #print(f'lower bound {lower_bound}')
                    #print(assble_q)
            # Linear interpolation between the previous and the current acceleration
            #print(assble_qdd_temp)
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
            # cost_total_intv_def = [('Regulated Energy', 'f8'), ('Number', 'i2')]
            # cost_total_intv = np.zeros((N), dtype = cost_total_intv_def)
  
            for i in range(N): # iterate through all N randomly generated accelerations
                #print(f'Debug: counter check, {i}')

                energy_total_intv = 0 # initialize total energy consumed by robot between time steps t1 and t2 (interval energy)
                # coeff = np.zeros(6)
                q_list = []
                qd_compare = []
                #print(N)
                #print(i)
                for j in range(6): # iterate through all joints, j = 0, ..., 5
                    qdd1 = result_qdd[sn, j] # start qdd for lerp
                    qdd2 = assble_qdd_temp[i, j] # end qdd for lerp
                    #print(assble_qdd)
                    #print(f'arguments for lerp: t1 = {t1}, t2 = {t2}, qdd1 = {qdd1}, qdd2 = {qdd2}\n')
                    qdd_sample[j, :] = [lerp_func(t1, t2, qdd1, qdd2, t) for t in t_val]
                    qd_sample[j, :] = [lerp_func_integral(t1, t2, qdd1, qdd2, t, result_qd[sn, j]) for t in t_val] # initial condition: qd0 = result_qd[sn, j], qd from previous time step
                    #qd_compare.append(simpson(qdd_sample[j, :], t_val))
                    q_sample[j, :] = [lerp_func_double_integral(t1, t2, qdd1, qdd2, t, result_qd[sn, j], result_q[sn, j]) for t in t_val] # qd0, q0 from result_qd, result_q
                    #for ii, t in enumerate(t_val):
                    #    index = np.where(np.round(joint[6], 2) == np.round(t, 2))
                        #print(np.round(t,2))
                        #print(index)
                        #print(joint[j].q[index[0]])
                    #    q_ref[j, ii] = joint[j].q[index[0]]
                    
                    #test_list = [t1, t2, qdd1, qdd2, result_q[sn, j], q_sample[j, :]]
                    #q_list.append(test_list)

                    """
                    lerp = interp1d(x, y)
                    time_sample = np.linspace(t1, t2, num=width)
                    qdd_sample[j, :] = lerp(time_sample)
                    qd_integral, error = quad(lerp, t1, t2)
                    qd_sample[j, :] = np.cumsum(qdd_sample[j, :]) * (time_sample[1] - time_sample[0])
                    #qd_sample[j, :] = [x+result_qd[sn, j] for x in qd_sample[j, :]]
                    q_sample[j, :] = np.cumsum(qd_sample[j, :]) * (time_sample[1] - time_sample[0])
                    #q_sample[j, :] = [x+result_q[sn, j] for x in q_sample[j, :]]
                    """
                
                cal_tau_qdd = np.transpose(qdd_sample) # qdd matrix for torque calculation, has a dimension of Width x 6
                #print(f'cal_tau_qdd preview: \n{cal_tau_qdd}\n')
                cal_tau_qd = np.transpose(qd_sample)
                #print(cal_tau_qdd)
                #print(cal_tau_qd)
                #print(joint[2].qd[20])
                assble_qd[i, :] = cal_tau_qd[-1, :]
                cal_tau_q = np.transpose(q_sample) # first row of q_sample looks like this [q0(t1), q0(t1+dt), q0(t1+2dt), ... , q0(t2)]
                q_opt = q_sample[:, -1]
                delta_q = q_opt - q_ref
                #print(f'check: {q_opt}')
                #print(qd_compare)
                #if sn >= 0 and i == 1:
                #    print(f'{q_list}\n\n')
                assble_q[i, :] = cal_tau_q[-1, :]
                #if i < 5:
                    #print(f'q_sample: \n{q_sample}')
                    #print(f'q_ref: \n{q_ref}')
                #    norm_sqr_diff = q_sample - q_ref
                #    print(f'element wise subtraction: \n{norm_sqr_diff}')
                energy_list = []
                power_list = []
                torque_list = []
                velocity_list = []
                #print(f'Debug counter check: {i}')

                for k in range(width): # iterate through all samples generated by lerp    
                    #print(f'Begin evaluation: {k+1}th row of the torque calculation matrices\n')    
                    #print(cal_tau_q)
                    #print(q_sample)
                    torq_vec = cal_tau(cal_tau_q[k, :], cal_tau_qd[k, :], cal_tau_qdd[k, :])
                    #torque_list.append(torq_vec)
                    #print(f'The calculated torque vector for all 6 joints: {torq_vec}\n') # reading the k-th row of the matrices for torque calculation, call the "cal_tau" function from call_tau.py
                    velocity_vec = cal_tau_qd[k, :]
                    #velocity_list.append(velocity_vec)
                    #print(f'The corresponding velocity vector for all 6 joints: {velocity_vec}\n')
                    power_val = abs(np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1)) # element-wise multiplication, then take 1-Norm
                    #power_list.append(power_val)
                    #print(f'Power Evaluation: {power_val}\n')
                    energy_total_intv = energy_total_intv + power_val * delta_t
                    #print(f'energy evaluation: {energy_total_intv}')
                    #energy_list.append(energy_total) # total energy for the first row of acceleration in assble_qdd
                    #print(f'Energy evaluation: {energy_total} with delta_t: {delta_t}\n')
                #print(f'Debug: {i}-th element of energy_val tuple is {energy_total_intv} for number {i}')
                energy_val[i] = (energy_total_intv, i)
                """
                sign_array = np.sign(delta_q) # recall that delta_q = q_opt - q_ref, we want to encourage q_opt > q_ref if qdd[0] > 0, vice versa
                for z in range(6):
                    if z == 1:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -50 if sign_array[z] > 0 else 50
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 50 if sign_array[z] < 0 else -50
                    elif z == 0 or z == 2 or z == 4:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -40 if sign_array[z] > 0 else 40
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 40 if sign_array[z] < 0 else -40
                    elif z == 5:
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -0.5 if sign_array[z] > 0 else 0
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 0 if sign_array[z] < 0 else -0.5
                    else: 
                        if u[z] == 1: # positive trapeze, q_opt > q_ref to be encouraged, coeff needs to be <0 to reduce cost function value
                            coeff[z] = -10 if sign_array[z] > 0 else 10
                        else: # negative trapeze, q_opt < q_ref to be encouraged
                            coeff[z] = 10 if sign_array[z] < 0 else -10
                #print(f'debug: penalty term{coeff * (sign_array * delta_q**2)}')
                cost_total_intv[i] = (energy_total_intv + (np.linalg.norm(coeff * (sign_array * delta_q**2), 1))**2, i)
                #print(energy_val[i])
                    #print(f'End evaluation: {k+1}th row of the torque calculation matrices\n')
                """
            #print(torque_list[-1])  
            #print(velocity_list[-1])
            #print(energy_val)
            sorted_energy_val = np.sort(energy_val, order='Energy')
            # sorted_cost_total = np.sort(cost_total_intv, order = 'Regulated Energy')
            #print(f'comparison: sorted cost \n{sorted_cost_total}')
            #print(f'comparison: sorted energy \n{sorted_energy_val}')
            #print(sorted_energy_val)
            post_q = np.zeros((Nbest, 6))
            post_qd = np.zeros((Nbest, 6))
            post_qdd = np.zeros((Nbest, 6))
            # num_array = sorted_cost_total['Number']
            num_array = sorted_energy_val['Number']
            #print(num_array)

            for i in range(Nbest):
                num = num_array[i] # returns the index for num-th best candidate
                post_q[i, :] = assble_q[num, :] # place the num th row of the assembled q-matrix onto the i-th row of the post_q matrix
                post_qd[i, :] = assble_qd[num, :]
                post_qdd[i, :] = assble_qdd_temp[num, :]
            #print(f'debug: post_qdd after one iteration: {post_qdd}')
            #print(post_qd)
            #print(post_q)


            mu_qdd_meas = np.mean(post_qdd, 0)
            var_qdd_post = np.var(post_qdd, 0)
            #print(f'post acceleration matrix post_qdd = \n{post_qdd}\nwith a variance of {var_qdd_post}\nmean = {mu_qdd_meas}\n\n')
            new_mu_sig_qdd = kalman_gain(sig_qdd, var_qdd_post, mu_qdd, mu_qdd_meas)
            mu_qdd_old = mu_qdd
            mu_qdd = new_mu_sig_qdd[0]
            sig_qdd = new_mu_sig_qdd[1]
            #print(f'new mean: {mu_qdd}')
            #print(f'new variance: {sig_qdd}')
            #print(iter)
            iter = iter + 1

        #print(f'post acceleration matrix post_qdd = \n{post_qdd}\nwith a variance of {var_qdd_post}\nmean = {mu_qdd_meas}\n\n')
        t_ref.append(t_val)
        #print(sorted_energy_val)
        #print(assble_qdd)
        #print(assble_q)
        result_q[sn+1, :] = assble_q[num_array[0], :]      
        #print(test_qd)
        result_qd[sn+1, :] = assble_qd[num_array[0], :]
        result_qdd[sn+1, :] = assble_qdd_temp[num_array[0], :]
        energy_total = energy_total + energy_val[num_array[0]]['Energy']

        print(f'Energy consumption converges at {assble_qdd_temp[num_array[0], :]} 1/s^2\n\nJoint config: {assble_q[num_array[0], :]} total energy consumption: {energy_val[num_array[0]]} J\n')
        #print(f'Optimized joint angle:\n{assble_q[0, :]}\n\nOptimized joint velocity:\n{assble_qd[0, :]}\n\nOptimized joint acceleration:\n{assble_qdd[0, :]}\n')
            #torque = cal_tau(assble_q[0, :], assble_qd[0, :], assble_qdd[0, :])
            #print(torque)
    """
    for i in range(6):
        end_q[i] = joint[i].q[-1]
        end_qd[i] = joint[i].qd[-1]
        end_qdd[i] = joint[i].qdd[-1]
    
    result_q = np.vstack((result_q, end_q))
    result_qd = np.vstack((result_qd, end_qd))
    result_qdd = np.vstack((result_qdd, end_qdd))
    time_vec[-1] = traj_time
    """

    print(result_q)
    print(result_qd)
    print(result_qdd)
    #print(t_ref)
    print(f'Optimization ended.\nOriginal energy consumption of the given trajectory is: {energy_og_total} J.\nTotal energy consumption of the optimizied trajectory is: {energy_total} J.\n')

    return result_q, result_qd, result_qdd, time_vec, joint

results = heuristic_kalman(50, 5, np.array([[0, 0, 0, 0, 0, 0], [0.0003, 5, 6, 8, 20, 10]]), 0, 0, 6, 3, 2)

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