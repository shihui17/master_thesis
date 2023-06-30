from math import pi, sqrt
import roboticstoolbox as rtb
import numpy as np
from traj import generate_traj_time
import matplotlib.pyplot as plt
import time as ti
from scipy.stats import truncnorm
from HKA_kalman_gain import *

Yu = rtb.models.DH.Yu()
case = 4
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
    
#traj = generate_traj_time(2, 201, start, end)
def get_contact_points():
    #points = [np.array([0, -0.13687, 0, 1]), np.array([-0.45, -0.05519, 0.12128, 1]), np.array([0, -0.05519, 0.12128, 1]), np.array([0, 0, 0.10946, 1]), 
            #np.array([0, 0, 0.10946, 1]), np.array([-0.05, 0, 0.07938, 1]), np.array([0, 0, 0, 1])]
    points = [np.array([0, -0.13687, 0, 1]), np.array([-0.44995, 0, 0.12128, 1]), np.array([0.01559, 0, 0.12128, 1]), np.array([0.05, 0, 0.07938, 1]), 
              np.array([0, -0.05, 0.07938, 1]), np.array([-0.05, 0, 0.07938, 1]), np.array([0, 0, 0, 1])]
    point_tuple_def = [('Point Coordinate', np.ndarray), ('Segment','i2')] # list that contains the energy consumption and the corresponding row index from assble_qdd
    point_tuple = np.zeros((7), dtype = point_tuple_def)
    #points = [np.array([0, -0.13687, 0, 1]), np.array([-0.50519, 0, 0.12128, 1]), np.array([-0.01559, 0.13687, 0, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0.10946, 1]), np.array([0, 0, 0, 1])] # points of interest
    for i, p in enumerate(points):
        point_tuple[i][0] = p

    point_tuple[0][1] = 0
    point_tuple[1][1] = 1
    point_tuple[2][1] = 1
    point_tuple[3][1] = 2
    point_tuple[4][1] = 3
    point_tuple[5][1] = 4
    point_tuple[6][1] = 5

    return point_tuple

def unit_vector(q, qd, point, joint_num): # angle and velocity matrix should have the size of (201, 6), point = np.array([x, y, z, 1]), t: traj time

    Yu = rtb.models.DH.Yu() # import Yu with all relevant geometric and mechanical data
    #traj = generate_traj_time(2.5, 201)

    trafo_p = np.eye(4)
    trafo_p[:, -1] = point
    joint_jacobian = np.zeros((6, 6))   
    pose_list = []
    poses = Yu.fkine_all(q)
    #print(poses)

    for pose in (poses):
        pose_list.append(np.array(pose))  

    rf_p =  pose_list[joint_num+1] @ trafo_p @ np.array([0, 0, 0, 1]) # Position vector of p in RF0

    for count in range(joint_num+1):
        translation = rf_p - pose_list[count] @ np.array([0, 0, 0, 1])
        translation = translation[:3]
        rotation = pose_list[count] @ np.array([0, 0, 1, 0])
        rotation = rotation[:3]
        joint_jacobian[:3, count] = np.cross(rotation, translation)
        joint_jacobian[3:6, count] = rotation

    lin_ang = joint_jacobian @ qd # velocity[t, :joint_num+1]
    lin_vel = lin_ang[:3]
    length = np.linalg.norm(lin_vel, 2)
    if length <= 1e-3:
        vel_u = np.zeros(3)
    else:
        vel_u = np.divide(lin_vel, (np.linalg.norm(lin_vel, 2)))


    fkine_ee = pose_list[-1]
    r6_0 = fkine_ee[:, -1]
    #print(rf_p)
    #print(r6_0)
    r6p_0 = rf_p - r6_0 # Position vector from RF6 to p in RF0
    #print(r6p_0)
    T_0e = Yu.fkine(q)
    T_0e = T_0e.A
    R_0e = T_0e[:3, :3]
    R_e0 = np.transpose(R_0e)
    #print(T_e0)
    r6p_0 = r6p_0[:3]
    r6p_0 = R_e0 @ r6p_0
    #print(r6p_0)
    trafo_6p = np.eye(4)
    trafo_6p[:3, -1] = r6p_0[:3]
    #print(trafo_6p)
    Yu.tool = trafo_6p
    jacobian_p = Yu.jacob0(q)
    #joint_jacobian = jacobian_p

    M_q = Yu.inertia(q)
    Aq_inv = joint_jacobian @ np.linalg.inv(M_q) @ np.transpose(joint_jacobian)

    if all(element == 0 for element in vel_u):
        m_refl = 0
        #print('oh no')
    else:
        m_refl = 1/(np.transpose(vel_u) @ Aq_inv[:3, :3] @ vel_u)

    return vel_u, m_refl, lin_vel, rf_p

def get_original_jointdata(trajectory_data):
    mass_vec = np.zeros(6)
    angle = np.zeros((6, 201))
    velocity = np.zeros((6, 201))
    accel = np.zeros((6, 201))
    q0 = np.zeros(6)
    qf = np.zeros(6)

    for j in range(6):
        angle[j, :] = trajectory_data[j].q
        velocity[j, :] = trajectory_data[j].qd
        accel[j, :] = trajectory_data[j].qdd
        mass_vec[j] = Yu[j].m
        q0[j] = angle[j, 0]
        qf[j] = angle[j, -1]
               
    mass = np.linalg.norm(mass_vec, 1) # total mass of robot
    return angle, velocity, accel, mass, q0, qf

def calculate_max_force(q_mat, qd_mat, contact_points_table, m_H, K, F_p):
    m_refl = np.zeros((7, 201))
    lin_vel = np.zeros((7, 3, 201))
    vel_scalar = np.zeros((7, 201))
    rf_p_storage = np.zeros((7, 3, 201))
    reduced_mass = np.zeros((7, 201))
    equi_contact_force = np.zeros((7, 201))

    for t in range(201):

        q = q_mat[:, t]
        qd = qd_mat[:, t]

        for n, point_list in enumerate(contact_points_table):
            #print(f'r = {r}\n')
            #print(q)
            result = unit_vector(q, qd, point_list[0], point_list[1])
            lin_vel[n, :, t] = result[2]
            vel_scalar[n, t] = np.linalg.norm(lin_vel[n, :, t], 2)
            m_refl[n, t] = result[1]
            if m_refl[n, t] == 0:
                reduced_mass[n, t] = 0
            else:
                reduced_mass[n, t] = 1 / (1 / m_refl[n, t] + 1 / m_H)
            equi_contact_force[n, t] = vel_scalar[n, t] * sqrt(reduced_mass[n, t] * K)
            rf_p_storage[n, :, t] = result[3][:3]

    #print(lin_vel)
    max_contact_force = np.amax(equi_contact_force, axis=1)
    max_of_all = np.amax(max_contact_force)
    index_max_of_all = np.argmax(max_of_all)
    delta_f = (max_of_all - F_p)**2

    return max_contact_force, max_of_all, delta_f, index_max_of_all

def hka_force_opt(N, Nbest, trajectory_data, start_joint_config, end_joint_config):
    """
    Randomly generate a set of trajectories and optimize for a most energy-efficient solution.
    Outputs the covariance matrix after each optimization iteration.
    Outputs the original and optimized energy consumption.
    Outputs algorithm runtime.
    Stores the optimized trajectory in prof_result_q.txt, prof_result_qd.txt and prof_result_qdd.txt, 

    :param N: size of the random trajectory set to be evaluated by HKA, 20 <= N <= 50 for optimal performance
    :param Nbest: the number of best sets of trajectory (i.e. Nbest candidates) to be considered by HKA
    :traj: original trajectory generated by tools.trapezoidal in roboticstoolbox
    :return result_q: the optimized angular trajectory
    :return result_qd: the optimized velocity trajectory
    :return result_qdd: the optimized acceleration trajectory
    """
    start_time = ti.time() # timer that records algorithm runtime
    vel_max = np.zeros(6)
    a = np.zeros(6)
    t_accel = np.zeros(6) 
    t_brake = np.zeros(6)
    t_min = np.zeros(6) # initialize minimal acceleration time, will later be derived from boundary conditions
    t_max = np.zeros(6) # initialize maximal acceleration time, will later be derived from boundary conditions
    #plot_trajectory(traj) # plot trajectory
    time = trajectory_data[6]
    step = trajectory_data[7] # total time step of the trajectory
    t0 = 0 # start
    tf = time[-1] # finish
    t_min = 1 * tf
    t_max = 4 * tf
    mu_t = np.zeros(6) # mean vector for Kalman
    sig_t =  np.zeros(6) # covariance vector for Kalman
    t_accel_rand = np.zeros((6, N)) # the matrix to store all N randomly generated acceleration time vectors for all 6 joints, hence the size 6 x N
    q0 = np.zeros(6)
    qf = np.zeros(6)
    q_mat = np.zeros((N, 6, step)) # matrix to store randomly generated angular trajectories of all 6 joints, N trajectories in total, hence the size N x 6 x step
    qd_mat = np.zeros((N, 6, step)) # randomly generated velocity trajectory
    qdd_mat = np.zeros((N, 6, step)) # randomly generated acceleration trajectory
    iter = 0 # iteration counter
    post_t_rand = np.zeros((6, Nbest)) # matrix to store the Nbest acceleration time vectors that yield the best energy efficiency
    tf_rand = np.zeros(N)
    contact_points_table = get_contact_points()
    original_jointdata =  get_original_jointdata(trajectory_data)
    angle = original_jointdata[0]
    velocity = original_jointdata[1]
    accel = original_jointdata[2]
    robot_mass = original_jointdata[3]
    q0 = original_jointdata[4]
    qf = original_jointdata[5]
    flag = np.full(6, False)
        
    K = 75 # Face is smashed against the robot, ouch!
    F_p = 65 # Face is smashed against the robot, ouch!
    m_H = 4.4 # Face is smashed against the robot, ouch!

    for j in range(6): # reads necessary data from the original trajectory
        angle[j, :] = trajectory_data[j].q
        velocity[j, :] = trajectory_data[j].qd
        accel[j, :] = trajectory_data[j].qdd
        q0[j] = angle[j, 0]
        qf[j] = angle[j, -1]

        if q0[j] == qf[j]: # if the joint does not move originally, then prevent the optimizer from moving it during optimization
            flag[j] = True

    original_data = calculate_max_force(angle, velocity, contact_points_table, m_H, K, F_p)
    original_force = original_data[1]
    max_force_point = original_data[3]

    if original_force < F_p:
        print(f'max force is {original_force} N, trajectory is safe, no optimization need!')
        return
    else:
        print(f'max force is {original_force} N, optimization is needed')
    mu_t = (t_max - t_min) / 2 # initialize mean vector
    sig_t = (t_max - t_min) / 3 # initialize std.dev.vector

    while iter <= 10:

        lb = (t_min - mu_t) / sig_t
        ub = (t_max - mu_t) / sig_t
        trunc_gen_t = truncnorm(lb, ub, loc=mu_t, scale=sig_t) 
        tf_rand = trunc_gen_t.rvs(size=N) # truncated gaussian distribution of size N is stored here 
        contact_force_list_def = [('Force','f8'), ('Number','i2'), ('Force Diff', 'f8'), ('index', 'i2')] # define a tuple that contains the energy consumption and the corresponding row index
        contact_force_list = np.zeros(N, dtype = contact_force_list_def) # initialize tuple

        for i in range(N): # iterate through trajectory set

            tg = generate_traj_time(tf_rand[i], step, start_joint_config, end_joint_config)
            for j in range(6):
                if flag[j] == False:
                    q_mat[i, j, :] = tg[j].q
                    #print(q_mat[i, j, :])
                    qd_mat[i, j, :] = tg[j].qd
                    qdd_mat[i, j, :] = tg[j].qdd
                else:
                    q_mat[i, j, :] = angle[j, :]
                    qd_mat[i, j, :] = velocity[j, :]
                    qdd_mat[i, j, :] = accel[j, :]

            force_data = calculate_max_force(q_mat[i, :, :], qd_mat[i, :, :], contact_points_table, m_H, K, F_p)
            max_contact_force = force_data[0]
            max_of_all = force_data[1]
            delta_f = force_data[2]
            contact_force_list[i] = (max_of_all, i, delta_f)
            print(contact_force_list)
            print(f'finished computation of set no. {i+1}')

        sorted_force_list = np.sort(contact_force_list, order='Force Diff') # sort energy consumption from lowest to highest
        print(sorted_force_list)
        num_array = sorted_force_list['Number'] # the corresponding indices
        t_rand_index = num_array[0 : Nbest] # the indices of the Nbest acceleration time vectors
        post_t_rand = [tf_rand[i] for i in t_rand_index]

        print(post_t_rand)

        mu_t_rand = np.mean(post_t_rand) # mean of Nbest candidates
        var_t_rand = np.var(post_t_rand) # variance of Nbest candidates
        new_mu_sig_t = kalman_gain(sig_t, var_t_rand, mu_t, mu_t_rand) # calculate Kalman gain, see HKA_kalman_gain.py
        mu_t = new_mu_sig_t[0] # new mean
        sig_t = new_mu_sig_t[1] # new std.dev.
        print(f'the diagonal of the covariance matrix:\n{var_t_rand}')

        if var_t_rand < 1e-6: # convergence criterion
            print(f'exited HKA at iter = {iter}')
            break
        
        print(f'End of iteration {iter}, begin iteration {iter+1}\n')
        iter = iter + 1
            #To plot search domain
            #if iter == 0:
            #    fig.suptitle(f'Search domain for Joint 1', fontsize=16)
            #    ax1.plot(time, q_mat[i, 0, :])
            #    ax1.set_xlabel('Travel Time in s')
            #    ax1.set_ylabel('Joint angle in rad')
            #    ax2.plot(time, qd_mat[i, 0, :])
            #    ax2.set_xlabel('Travel Time in s')
            #    ax2.set_ylabel('Joint velocity in rad/s')

    force_opt = sorted_force_list[num_array[0]] 
    result_q = q_mat[num_array[0], :, :]
    result_qd = qd_mat[num_array[0], :, :]
    result_qdd = qdd_mat[num_array[0], :, :]
    np.savetxt('force_result_q.txt', result_q)
    np.savetxt('force_result_qd.txt', result_qd)
    np.savetxt('force_result_qdd.txt', result_qdd)
    print(f'Original max contact force: ')
    print(f'Optimized total energy consumption: {force_opt} J')
    print(f'Optimization runtime: {ti.time() - start_time} seconds')
    print(f'Optimized initial acceleration: {result_qdd[:, 0]}')
    return result_q, result_qd, result_qdd   

def optimize_momentum(trajectory_data):

    m_refl = np.zeros((7, 201))
    lin_vel = np.zeros((7, 3, 201))
    vel_scalar = np.zeros((7, 201))
    rf_p_storage = np.zeros((7, 3, 201))
    contact_points_table = get_contact_points()
    original_jointdata =  get_original_jointdata(trajectory_data)
    angle = original_jointdata[0]
    velocity = original_jointdata[1]
    accel = original_jointdata[2]
    robot_mass = original_jointdata[3]
    reduced_mass = np.zeros((7, 201))
    equi_contact_force = np.zeros((7, 201))

    K = 75 # Face is smashed against the robot, ouch!
    F_p = 65 # Face is smashed against the robot, ouch!
    m_H = 4.4 # Face is smashed against the robot, ouch!


    for t in range(201):
        #if t == 1:
        #    print(q)
        q = angle[:, t]
        qd = velocity[:, t]

        for i, point_list in enumerate(contact_points_table):
            #print(f'r = {r}\n')
            result = unit_vector(q, qd, point_list[0], point_list[1])
            lin_vel[i, :, t] = result[2]
            vel_scalar[i, t] = np.linalg.norm(lin_vel[i, :, t], 2)
            m_refl[i, t] = result[1]
            if m_refl[i, t] == 0:
                reduced_mass[i, t] = 0
            else:
                reduced_mass[i, t] = 1 / (1 / m_refl[i, t] + 1 / m_H)
            equi_contact_force[i, t] = vel_scalar[i, t] * sqrt(reduced_mass[i, t] * K)
            rf_p_storage[i, :, t] = result[3][:3]

    print(equi_contact_force)
    print(m_refl)
    print(np.amax(equi_contact_force, axis = 1))
    print(np.amax(m_refl, axis = 1))
    np.savetxt("equivalent_force.txt", equi_contact_force)

start0 = np.array([0, 0, 0, 0, 0, 0])
end0 = np.array([-pi/2, 0, 0, -0, -0, 0])    

start1 = np.array([-pi, -pi/2, pi/2, -pi/2, -pi/2, 0])
end1 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start2 = np.array([pi/2, -pi/2, pi/2, -pi/2, -pi/2, 0])
end2 = np.array([pi, -pi, 0, pi/4, -pi/2, pi])

start3 = np.array([0, 0, 0, 0, 0, 0])
end3 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])

start4 = np.array([pi, -pi/2, pi/2, -pi/2, -pi/2, 0])
end4 = np.array([pi, -pi/3, pi/2, -5*pi/6, -0.58*pi, -0.082*pi])

start5 = np.array([0, -pi/2, pi/2, -pi/2, -pi/2, 0])
end5 = np.array([2*pi/3, -pi/8, pi, -pi/2, 0, -pi/3])

trajectory_data = generate_traj_time(0.8, 201, start1, end1)

hka_force_opt(10, 4, trajectory_data, start1, end1)
