'''
@author: Shihui Liu

A momentum-limited trajectory planning method using HKA and velocity profile optimization. Seven contact points are considered. For each contact point, the linear momentum over the entire trajectory
is calculated. The maximum linear momentum is determined. At last, the maximum linear momentum among all contact points is determined and optimized. 
'''

from math import pi
import roboticstoolbox as rtb
import numpy as np
from traj import generate_traj_time
import time as ti
from scipy.stats import truncnorm
from HKA_kalman_gain import *
import json
from calculate_momentum import *
from utility import *

Yu = rtb.models.DH.Yu() # import robot data

def hka_momentum_opt(N, Nbest, trajectory_data):
    """
    Randomly generate a set of trajectories and optimize the linear momentum of contact point using HKA.
    Outputs the covariance matrix after each optimization iteration.
    Outputs the original and optimized energy consumption.
    Outputs algorithm runtime.
    Stores the optimized trajectory in momentum_result_q.txt (and _qd, _qdd), stores the change of momentum after each HKA iteration in post_mean_momentum.txt
    Additionally stores all evaluated linear momentum of all contact points in momentum_history.json (requires post processing to make sense... Results are plotted and shown in master's thesis)

    :param N: size of the random trajectory set to be evaluated by HKA, 20 <= N <= 50 for optimal performance
    :param Nbest: the number of best sets of trajectory (i.e. Nbest candidates) to be considered by HKA
    :param trajectory_data: original trajectory obtained by get_original_jointdata
    :return result_q: the optimized angular trajectory
    :return result_qd: the optimized velocity trajectory
    :return result_qdd: the optimized acceleration trajectory
    """
    start_time = ti.time() # timer that records algorithm runtime
    vel_max = np.zeros(6) # initialise maximal coasting velocity
    a = np.zeros(6) # initial acceleration of each joint
    t_accel = np.zeros(6) # initialise acceleration time
    t_brake = np.zeros(6) # initial brake time
    t_min = np.zeros(6) # initialize minimal acceleration time, will later be derived from boundary conditions
    t_max = np.zeros(6) # initialize maximal acceleration time, will later be derived from boundary conditions
    #plot_trajectory(traj) # plot trajectory
    time = trajectory_data[6] # discretised trajectory time (with 200 increments)
    step = trajectory_data[7] # total time step of the trajectory
    t0 = 0 # start
    tf = time[-1] # finish, length of trajectory time
    mu_t = np.zeros(6) # mean vector for Kalman
    sig_t =  np.zeros(6) # covariance vector for Kalman
    t_accel_rand = np.zeros((6, N)) # the matrix to store all N randomly generated acceleration time vectors for all 6 joints, hence the size 6 x N
    q0 = np.zeros(6) # initial joint config
    qf = np.zeros(6) # final joint config
    q_mat = np.zeros((N, 6, step)) # matrix to store randomly generated angular trajectories of all 6 joints, N trajectories in total, hence the size N x 6 x step
    qd_mat = np.zeros((N, 6, step)) # randomly generated velocity trajectory
    qdd_mat = np.zeros((N, 6, step)) # randomly generated acceleration trajectory
    iter = 0 # iteration counter
    post_t_rand = np.zeros((6, Nbest)) # matrix to store the Nbest acceleration time vectors that yield the best energy efficiency

    m_refl_og = np.zeros((7, 201)) # original reflected mass
    lin_vel_og = np.zeros((7, 3, 201)) # original linear velocity of contact point
    vel_scalar_og = np.zeros((7, 201)) # original scalar velocity of contact point
    m_refl = np.zeros((7, 201)) # optimized reflected mass
    lin_vel = np.zeros((7, 3, 201)) # optimized linear velocity of contact point
    vel_scalar = np.zeros((7, 201)) # optimized scalar velocity of contact point
    rf_p_storage = np.zeros((7, 3, 201)) # for plotting, not used anymore
    contact_points_table = get_contact_points() # see utility.py
    original_jointdata =  get_original_jointdata(trajectory_data) # see utility.py
    angle = original_jointdata[0]
    velocity = original_jointdata[1]
    q0 = original_jointdata[4]
    qf = original_jointdata[5]
    momentum_history = [] # initialise list for json file
    momentum_per_iter = np.zeros((N, 7)) # for printing intermediate results
    reduced_mass = np.zeros((7, 201)) # used for other optimization methods
    equi_contact_force = np.zeros((7, 201)) # used for other optimization methods
    post_mean_momentum = [] # the optimized momentum of all contact points after each HKA iteration
    K = 35000 # Spring constant of a human shoulder
    F_p = 210 # Maximal permitted force for a human shoulder
    m_H = 40 # Effective mass of a human shoulder

    for t in range(201): # iterate through all trajectory time steps

        q = angle[:, t] # read instantaneous angle
        qd = velocity[:, t] # read instantaneous velocity

        for n, point_list in enumerate(contact_points_table): # iterate through table of contact points
            result = calculate_momentum(q, qd, point_list[0], point_list[1]) # see calculate_momentum.py
            lin_vel_og[n, :, t] = result[2] # original linear velocity of the contact point
            vel_scalar_og[n, t] = np.linalg.norm(lin_vel_og[n, :, t], 2) # scalar velocity
            m_refl_og[n, t] = result[1] # original reflected mass of the contact point

    momentum_og = np.multiply(vel_scalar_og, m_refl_og) # calulcate momentum
    max_momentum_og = np.amax(momentum_og, axis=1) # maximum momentum among all contact points based on original trajectory
    print(max_momentum_og)

    for j in range(6): # define boundary conditions, similar to 03_Profile_Optimization

        if q0[j] == qf[j]: # if the joint does not move originally, then prevent the optimizer from moving it during optimization
            vel_max[j] = 0
            t_accel[j] = 0
            t_brake[j] = tf
            a[j] = 0
            t_min[j] = abs(vel_max[j] / 100)
            t_max[j] = abs(tf / 3 + vel_max[j] / 100)
        else:
            vel_max[j] = (qf[j] - q0[j]) / tf * 1.5 
            t_accel[j] = np.round((q0[j] - qf[j] + vel_max[j] * tf) / vel_max[j], 2) # end of acceleration, rounded to 2 decimals to exactly match the time points in traj[6]
            t_brake[j] = np.round(tf - t_accel[j], 2) # start of braking
            a[j] = vel_max[j] / t_accel[j]
            t_min[j] = abs(vel_max[j] / 100) # boundary condition: maximal angular acceleration 100 s^-2
            t_max[j] = abs(tf - tf / 3 - vel_max[j] / 100) # maximal allowed acceleration time is reached when the joint has to brake with the maximal angular acceleration in order to reach end configuration (-100 s^-2)

    mu_t = t_accel # initialize mean vector
    sig_t = (t_max - t_min) / 2 # initialize std. dev. vector

    while iter <= 15: # begin HKA iteration

        for j in range(6): # generate truncated Gaussian distribution to account for boundary conditions
            lb = (t_min[j] - mu_t[j]) / sig_t[j]
            ub = (t_max[j] - mu_t[j]) / sig_t[j]
            trunc_gen_t = truncnorm(lb, ub, loc=mu_t[j], scale=sig_t[j]) 
            t_accel_rand[j, :] = trunc_gen_t.rvs(size=N) # random acceleration time
          
        max_momentum_list_def = [('Momentum','f8'), ('Number','i2'), ('Point', 'i2')] # define a tuple that contains momentum, its index and the corresponding contact point
        max_momentum_list = np.zeros((N), dtype = max_momentum_list_def) # initialize tuple

        for i in range(N): # iterate through N randomly generated trajectory
            for j in range(6): # iterate through each joint
                max_velocity = vel_max[j]
                t_accel_r = t_accel_rand[j, i] # randomly generated acceleration time
                q0_r = q0[j] # initial angle of the joint j
                qf_r = qf[j] # final angle of the joint j
                tg = gen_traj(max_velocity, t_accel_r, q0_r, qf_r, tf, time) # generate trajectory based on t_accel_r, see random_traj.py
                q_mat[i, j, :] = tg[0] # randomly generated angle trajectory for joint j, based on t_accel_r
                qd_mat[i, j, :] = tg[1] # randomly generated velocity trajectory for joint j, based on t_accel_r

            for t in range(201): # iterate through all trajectory time steps

                q = q_mat[i, :, t] # angle of each joint at time t, based on the randomly generated angle trajectory
                qd = qd_mat[i, :, t] # velocity of each joint at time t, based on the randomly generated velocity trajectory

                for n, point_list in enumerate(contact_points_table):
                    result = calculate_momentum(q, qd, point_list[0], point_list[1]) # calculate momentum, see # calculate_momentum.py
                    lin_vel[n, :, t] = result[2]
                    vel_scalar[n, t] = np.linalg.norm(lin_vel[n, :, t], 2)
                    m_refl[n, t] = result[1]

            momentum = np.multiply(vel_scalar, m_refl) # linear momentum of all contact points
            max_momentum = np.amax(momentum, axis=1) # maximal momentum of all contact points over the trajectory, size: 6x1
            momentum_per_iter[i, :] = max_momentum # store the maximal momentum of all contact points to write into json file later
            max_of_all = np.amax(max_momentum) # the maximum of the maximal momentum of all contact points, size: 1x1
            index_max_of_all = np.argmax(max_momentum) # the index of the corresponding contact point
            max_momentum_list[i] = (max_of_all, i, index_max_of_all) # store maximum momentum, index of random sample, index of contact point
            print(max_momentum_list) 
            print(f'finished computation of set no. {i+1}')

        sorted_momentum_list = np.sort(max_momentum_list, order='Momentum') # sort momentum from lowest to highest
        num_array = sorted_momentum_list['Number'] # the corresponding indices
        t_rand_index = num_array[0 : Nbest] # the indices of the Nbest acceleration time vectors
        momentum_history.append(momentum_per_iter.tolist()) # convert to list, otherwise incompatible with json
        post_mean_momentum.append(np.mean(sorted_momentum_list[0][0])) # store the new momentum of each contact point

        for j in range(6):
            post_t_rand[j, :] = [t_accel_rand[j, i] for i in t_rand_index] # store accel time vectors into a big matrix to run through HKA

        mu_t_rand = np.mean(post_t_rand, 1) # mean of Nbest candidates
        var_t_rand = np.var(post_t_rand, 1) # variance of Nbest candidates
        new_mu_sig_t = kalman_gain(sig_t, var_t_rand, mu_t, mu_t_rand) # calculate Kalman gain, see HKA_kalman_gain.py
        mu_t = new_mu_sig_t[0] # new mean
        sig_t = new_mu_sig_t[1] # new std.dev.
        print(f'the diagonal of the covariance matrix:\n{var_t_rand}')

        if (all(i < 1e-4 for i in var_t_rand) == True) or (max_of_all < 1): # convergence criterion
            print(f'exited HKA at iter = {iter}')
            break
        
        print(f'End of iteration {iter}, begin iteration {iter+1}\n')
        iter = iter + 1

    momentum_opt = max_momentum_list[num_array[0]] 
    result_q = q_mat[num_array[0], :, :]
    result_qd = qd_mat[num_array[0], :, :]
    result_qdd = qdd_mat[num_array[0], :, :]
    np.savetxt('momentum_result_q.txt', result_q)
    np.savetxt('momentum_result_qd.txt', result_qd)
    np.savetxt('momentum_result_qdd.txt', result_qdd)
    np.savetxt('post_mean_momentum.txt', post_mean_momentum)

    with open('momentum_history.json', 'w') as file:
        json.dump(momentum_history, file)

    print(f'Original maximal momentum: {max_momentum_og} kg*m/s')
    print(f'Optimized maximal momentum: {momentum_opt} kg*m/s')
    print(f'Optimization runtime: {ti.time() - start_time} seconds')
    print(f'Optimized initial acceleration: {result_qdd[:, 0]}')
    return result_q, result_qd, result_qdd   

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

trajectory_data = generate_traj_time(2, 201, start1, end1)

hka_momentum_opt(10, 3, trajectory_data)

"""
case = 
if case == 0: # zero-pose, robot rotates around axis 1 for pi/2
    start = np.array([0, 0, 0, 0, 0, 0])
    end = np.array([-pi/2, 0, 0, -0, -0, 0])    
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.44995, 0, 0.12128, 1]), np.array([0.01559, 0, 0.12128, 1]), np.array([0.05, 0, 0.07938, 1]), 
              np.array([0, -0.05, 0.07938, 1]), np.array([-0.05, 0, 0.07938, 1]), np.array([0, 0, 0, 1])]
elif case == 1: # reverse of case 0
    start = np.array([-0, 0, 0, 0, 0, 0])
    end = np.array([pi/2, 0, 0, -0, -0, 0])
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.50519, 0, 0.12128, 1]), np.array([0, 0, 0.13687, 1]), np.array([0, 0, 0.10946, 1]), 
            np.array([0, 0.05, 0.07938, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0, 1])]
elif case == 2: 
    start = np.array([-0, -pi/2, 0, 0, 0, 0])
    end = np.array([0, 0, 0, -0, -0, 0])
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.50519, 0, 0.12128, 1]), np.array([0, 0.01559, 0.12128, 1]), np.array([0, 0.05, 0.07938, 1]), 
              np.array([0, 0, 0.10946, 1]), np.array([0, 0.05, 0.07938, 1]), np.array([0, 0, 0, 1])]
elif case == 3:
    start = np.array([-0, -pi/2, 0, 0, 0, 0])
    end = np.array([0, -pi, 0, -0, -0, 0])
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.50519, 0, 0.12128, 1]), np.array([-0.01559, 0.13687, 0, 1]), np.array([0, 0, 0.10946, 1]), 
            np.array([0, -0.05, 0.07938, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0, 1])]
elif case == 4:
    #start = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
    #end = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])
    start = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
    end = np.array([pi/2, -pi/2, pi/2, -pi/2, -pi/2, 0])
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.45, -0.05519, 0.12128, 1]), np.array([0, -0.05519, 0.12128, 1]), np.array([0, 0, 0.10946, 1]), 
            np.array([0, 0, 0.10946, 1]), np.array([-0.05, 0, 0.07938, 1]), np.array([0, 0, 0, 1])]
"""