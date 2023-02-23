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
    max_iter = 100
    lb = matlib.repmat(D[0, :], N, 1)
    ub = matlib.repmat(D[1, :], N, 1)
    assble_q = np.zeros((N, n))
    assble_qd = np.zeros((N, n))
    assble_qdd = np.zeros((N, n))
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
    time_vec = np.zeros(sample_num+2)
    ctr = 0
    dt = joint[6][1] - joint[6][0]

    for i in range(n): # write initial angle, velocity and acceleration for all joints in the first rows of the result lists
        result_q[0, i] = joint[i].q[0]
        result_qd[0, i] = joint[i].qd[0]
        result_qdd[0, i] = joint[i].qdd[0]
    #print(result_qdd)

    for sn in range(sample_num): # now iterate through all sample points
        energy_og = 0 # initialize pre-optimization energy consumption
        iter = 0 # initialize iteration counter
        time_vec[sn+1] = traj_time/(sample_num+1)*(sn+1) # record time steps in time vector, time_vec[0] = 0
        print(time_vec)
        mu_index_pre = math.floor(100/(sample_num+1)*sn)
        mu_index = math.floor(100/(sample_num+1)*(sn+1))
        #print(f'TEST!: {time_vec}')
        #print(f'Calculating optimized trajectory at sample point {sn}, corresponding to trajectory time {traj_steps/(sample_num+1)*(sn+1)}\n')

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

        # initialize mean vector mu_q, mu_qd, mu_qdd, and std vector sig_qdd
        for i in range(n):
            
            mu_q[i] = joint[i].q[mu_index]
            mu_qd[i] = joint[i].qd[mu_index]
            mu_qdd[i] = joint[i].qdd[mu_index]
            #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
            if mu_qdd[i] == 0:
                sig_qdd[i] = 0.02 # initialize std vector, hard coded for now
            else:
                sig_qdd[i] = abs(0.1*np.sqrt(abs(mu_qdd[i])))
            
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
            
            for i in range(6): # generate gaussian destribution for acceleration of each joint (N random values), results written in assble_qdd (an Nx6 matrix)
                if mu_qdd[i] < 0:
                    lb = -6
                    ub = (0 - mu_qdd[i]) / sig_qdd[i]
                    s_qdd_trunc = truncnorm(lb, ub, loc=mu_qdd[i], scale=sig_qdd[i])
                    assble_qdd[:, i] = np.transpose(s_qdd_trunc.rvs(size=N))
                    a = 0
                    #print(a)
                elif mu_qdd[i] > 0:
                    lb = (0 - mu_qdd[i]) / sig_qdd[i]
                    ub = 6
                    s_qdd_trunc = truncnorm(lb, ub, loc=mu_qdd[i], scale=sig_qdd[i])
                    assble_qdd[:, i] = np.transpose(s_qdd_trunc.rvs(size=N))
                    a = 1
                    #print(a)
                else:
                    s_qdd = np.random.normal(0, sig_qdd[i], N)
                    assble_qdd[:, i] = np.transpose(s_qdd)
                    a = 2
                    #print(a)
            #print(a)
            #print(len(assble_qdd))
            #print(assble_qdd)
            #print(result_q)

            # Linear interpolation between the previous and the current acceleration
            t1 = time_vec[sn] # we are at sample point sn+1 right now, meaning previous sample point is sn         
            t2 = time_vec[sn+1] # current sample point
            width = 10
            qdd_sample = np.zeros((6, width))
            qd_sample = np.zeros((6, width))
            q_sample = np.zeros((6, width))
            t_val = np.linspace(t1, t2, num = width)
            #print(t_val)
            delta_t = t_val[1] - t_val[0]
            power = np.zeros((N, 6))

            energy_val_def = [('Energy','f8'), ('Number','i2')] # list that contains the energy consumption and the corresponding row index from assble_qdd
            energy_val = np.zeros((N), dtype = energy_val_def)


            for i in range(N): # iterate through all N randomly generated accelerations

                energy_total = 0 # initialize total energy consumed by robot between time steps t1 and t2
                q_list = []
                qd_compare = []
                #print(N)
                #print(i)
                for j in range(6): # iterate through all joints, j = 0, ..., 5
                    qdd1 = result_qdd[sn, j] # start qdd for lerp
                    qdd2 = assble_qdd[i, j] # end qdd for lerp
                    qd1 = result_qd[sn, j]
                    qd2 = assble_qd[i, j]
                    #print(assble_qdd)
                    #print(f'arguments for lerp: t1 = {t1}, t2 = {t2}, qdd1 = {qdd1}, qdd2 = {qdd2}\n')
                    qdd_sample[j, :] = [lerp_func(t1, t2, qdd1, qdd2, t) for t in t_val]
                    qd_sample[j, :] = [lerp_func_integral(t1, t2, qdd1, qdd2, t, result_qd[sn, j]) for t in t_val] # initial condition: qd0 = result_qd[sn, j], qd from previous time step
                    #qd_compare.append(simpson(qdd_sample[j, :], t_val))
                    q_sample[j, :] = [lerp_func_double_integral(t1, t2, qdd1, qdd2, t, result_qd[sn, j], result_q[sn, j]) for t in t_val] # initial condition, q0 = result_q[sn, j], q from previous time step
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
                cal_tau_q = np.transpose(q_sample)
                #print(qd_compare)
                #if sn >= 0 and i == 1:
                #    print(f'{q_list}\n\n')
                assble_q[i, :] = cal_tau_q[-1, :]
                energy_list = []
                power_list = []
                torque_list = []
                velocity_list = []

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
                    power_val = np.linalg.norm(np.multiply(torq_vec, velocity_vec), 1) # element-wise multiplication, then take 1-Norm
                    #power_list.append(power_val)
                    #print(f'Power Evaluation: {power_val}\n')
                    energy_total = energy_total + power_val * delta_t
                    #energy_list.append(energy_total) # total energy for the first row of acceleration in assble_qdd
                    #print(f'Energy evaluation: {energy_total} with delta_t: {delta_t}\n')
                    energy_val[i] = (energy_total, i) 
                    #print(f'End evaluation: {k+1}th row of the torque calculation matrices\n')
                
            #print(torque_list[-1])  
            #print(velocity_list[-1])
                
            sorted_energy_val = np.sort(energy_val, order='Energy')
            #print(sorted_energy_val)
            post_q = np.zeros((Nbest, 6))
            post_qd = np.zeros((Nbest, 6))
            post_qdd = np.zeros((Nbest, 6))
            num_array = sorted_energy_val['Number']
            #print(num_array)

            for i in range(Nbest):
                num = num_array[i] # returns the index for num-th best candidate
                post_q[i, :] = assble_q[num, :] # place the num th row of the assembled q-matrix onto the i-th row of the post_q matrix
                post_qd[i, :] = assble_qd[num, :]
                post_qdd[i, :] = assble_qdd[num, :]
            #print(post_qdd)
            #print(post_qd)
            #print(post_q)


            mu_qdd_meas = np.mean(post_qdd, 0)
            var_qdd_post = np.var(post_qdd, 0)
            #print(f'post acceleration matrix post_qdd = {post_qdd}\nwith a variance of {var_qdd_post}\nStd.Dev = {sig_qdd}\n\n')
            new_mu_sig_qdd = kalman_gain(sig_qdd, var_qdd_post, mu_qdd, mu_qdd_meas)
            mu_qdd_old = mu_qdd
            mu_qdd = new_mu_sig_qdd[0]
            sig_qdd = new_mu_sig_qdd[1]
            #print(f'new mean: {mu_qdd}')
            #print(f'new variance: {sig_qdd}')
            iter = iter + 1

        #print(sorted_energy_val)

        #print(assble_qdd)
        #print(assble_q)
        result_q[sn+1, :] = assble_q[num_array[0], :]

        
        #print(test_qd)
        result_qd[sn+1, :] = assble_qd[num_array[0], :]
        result_qdd[sn+1, :] = assble_qdd[num_array[0], :]



        print(f'Power output of each joint converges at {assble_qdd[0, :]} 1/s^2, total power output: {sorted_energy_val[0]} J\n')
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

    return result_q, result_qd, result_qdd, time_vec, joint

results = heuristic_kalman(50, 5, np.array([[0, 0, 0, 0, 0, 0], [0.0003, 5, 6, 8, 20, 10]]), 0, 0, 6, 4, 2)
joint = results[4]
time = results[3][0:5]
q = results[0]
qd = results[1]
qdd = results[2]
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(joint[6], joint[2].q)
ax1.plot(time, q[:, 2], 'r+')
ax2.plot(joint[6], joint[2].qd)
ax2.plot(time, qd[:, 2], 'r+')
ax3.plot(joint[6], joint[2].qdd)
ax3.plot(time, qdd[:, 2], 'r+')

plt.show()

"""
            power = np.zeros((N, 6))
            val_func_def = [('Power','f8'), ('Number','i2')]
            val_func = np.zeros((N), dtype = val_func_def)
            for i in range(N):
                torq_vec = np.abs(cal_tau(assble_q[i, :], assble_qd[i, :], assble_qdd[i, :]))
                velocity_vec = assble_qd[i, :]
                power[i, :] = np.abs(np.multiply(torq_vec, velocity_vec))
                val_func[i] = (np.linalg.norm(power[i, :], 1), i+1) 
            #print(f'The power output for each joint with {N} samples is :\n{power}\n')
            #print(f'The total power output of the robot (col.1) with the corresponding sample number (col.2) is:\n{val_func}\n')
            #print(f'The assembled joint angle matrix for all 6 joints with {N} samples is:\n {assble_q}\n')
            #print(f'The assembled joint velocity matrix for all 6 joints with {N} samples is:\n {assble_qd}\n')
            #print(f'The assembled joint acceleration matrix for all 6 joints with {N} samples is:\n {assble_qdd}\n')

            #print(val_func)
            sort_val_func = np.sort(val_func, order='Power')
            #print(f'{sort_val_func}\nIteration: {iter}\n') # uncomment to print cost function value in each iteration

            post_q = np.zeros((Nbest, 6))
            post_qd = np.zeros((Nbest, 6))
            post_qdd = np.zeros((Nbest, 6))
            num_array = sort_val_func['Number']
            #print(num_array)
            #print(num_array[0])
            #print(assble_q)
            #print(post_q)

            # assemble sorted q, qd, and qdd matrices for update step
            for i in range(Nbest):
                num = num_array[i] # returns the index for num-th best candidate
                post_q[i, :] = assble_q[num-1, :] # place the (num-1)th row of the assembled q-matrix onto the i-th row of the post_q matrix
                post_qd[i, :] = assble_qd[num-1, :]
                post_qdd[i, :] = assble_qdd[num-1, :]

            #print(f'POST {post_q}')
            mu_q_meas = np.mean(post_q, 0)
            mu_qd_meas = np.mean(post_qd, 0)
            mu_qdd_meas = np.mean(post_qdd, 0)
            var_q_post = np.var(post_q, 0)
            var_qd_post = np.var(post_qd, 0)
            var_qdd_post = np.var(post_qdd, 0)
            
            
            #Compute Kalman gain for q, qd, qdd
            new_mu_sig_q = kalman_gain(sig_q, var_q_post, mu_q, mu_q_meas)
            mu_q_old = mu_q
            mu_q = new_mu_sig_q[0]
            sig_q = new_mu_sig_q[1]

            new_mu_sig_qd = kalman_gain(sig_qd, var_qd_post, mu_qd, mu_qd_meas)
            mu_qd_old = mu_qd
            mu_qd = new_mu_sig_qd[0]
            sig_qd = new_mu_sig_qd[1]

            new_mu_sig_qdd = kalman_gain(sig_qdd, var_qdd_post, mu_qdd, mu_qdd_meas)
            mu_qdd_old = mu_qdd
            mu_qdd = new_mu_sig_qdd[0]
            sig_qdd = new_mu_sig_qdd[1]

            #print(mu_q)
            #print(f'old mean vector for q:\n{mu_q_old}\nfor qd:\n{mu_qd_old}\n:for qdd:\n{mu_qdd_old}\n')
            """
"""
            L = np.divide(np.square(sig_q), (np.square(sig_q) + var_q_post))
            mu_q_post = mu_q + np.multiply(L, (mu_q_meas - mu_q))
            P = np.square(sig_q) - np.multiply(L, np.square(sig_q))
            #print(P)
            #print(var_q_post)
            sig_q = sig_q + (np.sqrt(P) - sig_q)
            mu_q = mu_q_post
            
            #print(assble_qd)-
        
            iter = iter + 1
        
        result_q[sn+1, :] = assble_q[0, :]
        result_qd[sn+1, :] = assble_qd[0, :]
        result_qdd[sn+1, :] = assble_qdd[0, :]
        print(f'Power output of each joint converges at {assble_qd[0, :]} rad/s, total power output: {sort_val_func[0]} W\n')
        print(f'Optimized joint angle:\n{assble_q[0, :]}\n\nOptimized joint velocity:\n{assble_qd[0, :]}\n\nOptimized joint acceleration:\n{assble_qdd[0, :]}\n')

    for i in range(6):
        end_q[i] = joint[i].q[-1]
        end_qd[i] = joint[i].qd[-1]
        end_qdd[i] = joint[i].qdd[-1]
    
    result_q = np.vstack((result_q, end_q))
    result_qd = np.vstack((result_qd, end_qd))
    result_qdd = np.vstack((result_qdd, end_qdd))
    time_vec[-1] = traj_steps

    return result_q, result_qd, result_qdd, time_vec, joint
    """


