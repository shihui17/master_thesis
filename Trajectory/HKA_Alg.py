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
from quintic_hermite import *
from scipy.interpolate import CubicHermiteSpline, BPoly

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
    L = np.divide(np.square(sig), (np.square(sig) + var_post)) # Kalman gain
    mu_post = mu + np.multiply(L, (mu_meas - mu)) # post mean
    P = np.square(sig) - np.multiply(L, np.square(sig)) # post covariance
    s = np.min([1, np.mean(np.sqrt(var_post))])
    a = 0.6 * (s**2 / (s**2 + np.max(P))) # slowdown factor to slow down the convergence, coefficient (in this case 0.6) can vary (see HKA book)
    #print(P)
    #print(var_q_post)
    sig_new = sig + a * (np.sqrt(P) - sig) # set new std. deviation
    mu_new = mu_post # set new mean
    return mu_new, sig_new

def heuristic_kalman(N, Nbest, D, alpha, sd, n, sample_num, traj_steps):
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
    max_iter = 20
    lb = matlib.repmat(D[0, :], N, 1)
    ub = matlib.repmat(D[1, :], N, 1)
    assble_q = np.zeros((N, n))
    assble_qd = np.zeros((N, n))
    assble_qdd = np.zeros((N, n))
    result_q = np.zeros((sample_num+1, 6))
    result_qd = np.zeros((sample_num+1, 6))
    result_qdd = np.zeros((sample_num+1, 6))
    end_q = np.zeros(6)
    end_qd = np.zeros(6)
    end_qdd = np.zeros(6)
    debug = 0
    joint = generate_traj(traj_steps) # generate trapezoidal trajectory with 20 time steps, change config in traj.py
    time_vec = np.zeros((sample_num+2, 1))
    for sn in range(sample_num):
        iter = 0
        time_vec[sn+1] = traj_steps/(sample_num+1)*(sn+1)
        print(f'TEST!: {time_vec}')
        print(f'Calculating optimized trajectory at sample point {sn}, corresponding to trajectory time {traj_steps/(sample_num+1)*(sn+1)}\n')

        for i in range(n):
            result_q[0, i] = joint[i].q[0]
            result_qd[0, i] = joint[i].qd[0]
            result_qdd[0, i] = joint[i].qdd[0]

        # initialize mean vector mu_q, mu_qd, mu_qdd, and std vector sig_q, sig_qd, sig_qdd
        for i in range(n):
            mu_q[i] = joint[i].q[math.floor(traj_steps/(sample_num+1)*(sn+1))]
            #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
            sig_q[i] = abs(0.05*mu_q[i]+0.00001) # initialize std vector
            mu_qd[i] = joint[i].qd[math.floor(traj_steps/(sample_num+1)*(sn+1))]
            #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
            sig_qd[i] = abs(0.05*mu_qd[i]+0.00001) # initialize std vector
            mu_qdd[i] = joint[i].qdd[math.floor(traj_steps/(sample_num+1)*(sn+1))]
            #mu[i] = (D[0, i] + D[1, i])/2 # initialize mean vector
            sig_qdd[i] = abs(0.05*mu_qdd[i]+0.00001) # initialize std vector

        print(f'Original joint angle vector:\n{mu_q}\n')
        print(f'Original joint velocity vector:\n{mu_qd}\n')
        print(f'Original joint acceleration vector:\n{mu_qdd}\n')

        ref_torq = cal_tau(mu_q, mu_qd, mu_qdd) # original output torque for reference
        ref_power = np.multiply(ref_torq, mu_qd) # calculate orginal power output of each joint
        ref_total_power = np.linalg.norm(ref_power, 1) # summation to get total power output
        print(f'Original total power output: {ref_total_power} W\n')

        while iter <= max_iter:
            for i in range (n):
                s_q = np.random.normal(mu_q[i], sig_q[i], N)
                assble_q[:, i] = np.transpose(s_q)
                s_qd = np.random.normal(mu_qd[i], sig_qd[i], N)
                assble_qd[:, i] = np.transpose(s_qd)
                s_qdd = np.random.normal(mu_qdd[i], sig_qdd[i], N)
                assble_qdd[:, i] = np.transpose(s_qdd)
            ref_qd = assble_qd
            #torque = cal_tau(assble_q[0, :], assble_qd[0, :], assble_qdd[0, :])
            #print(torque)
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
            L = np.divide(np.square(sig_q), (np.square(sig_q) + var_q_post))
            mu_q_post = mu_q + np.multiply(L, (mu_q_meas - mu_q))
            P = np.square(sig_q) - np.multiply(L, np.square(sig_q))
            #print(P)
            #print(var_q_post)
            sig_q = sig_q + (np.sqrt(P) - sig_q)
            mu_q = mu_q_post
            """
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


def evaluate_plot(joint_num):
    result_assemble = heuristic_kalman(50, 5, np.array([[0, 0, 0, 0, 0, 0], [0.0003, 5, 6, 8, 20, 10]]), 0, 0, 6, 6, 20)
    result_q = result_assemble[0]
    result_qd = result_assemble[1]
    result_qdd = result_assemble[2]
    time_vec = result_assemble[3]
    matrix_eval = np.zeros((len(result_q), 3))
    for i in range (len(result_q)):
        matrix_eval[i, 0] = result_q[i, joint_num]
        matrix_eval[i, 1] = result_qd[i, joint_num]
        matrix_eval[i, 2] = result_qdd[i, joint_num]

    print(matrix_eval)

    joint_data = result_assemble[4]
    q_plot = joint_data[joint_num].q
    qd_plot = joint_data[joint_num].qd
    qdd_plot = joint_data[joint_num].qdd
    tt = joint_data[6]

    approx_func = quintic_hermite_approx(len(result_q), matrix_eval, time_vec, 100)[0]
    time = quintic_hermite_approx(len(result_q), matrix_eval, time_vec, 100)[1]
    approx_func_d = quintic_hermite_approx(len(result_q), matrix_eval, time_vec, 100)[2]
    approx_func_dd = quintic_hermite_approx(len(result_q), matrix_eval, time_vec, 100)[3]
    approx_func = np.concatenate(approx_func)
    approx_func_d = np.concatenate(approx_func_d)
    approx_func_dd = np.concatenate(approx_func_dd)
    time = np.concatenate(time)
    print(approx_func)
    print(approx_func_d)
    print(approx_func_dd)
    print(time)
    #print(time_vec)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained',)
    ax1.plot(time, approx_func, color='red')
    ax2.plot(time, approx_func_d)
    ax3.plot(time, approx_func_dd)

    ax1.plot(tt*4, q_plot, color='blue')
    ax2.plot(tt*4, qd_plot)
    ax3.plot(tt*4, qdd_plot)
    #print(q_plot)
    #print(tt)
    plt.show()


evaluate_plot(2)

result_assemble = HKA(20, 0, np.array([[0, 0, 0, 0, 0, 0], [0.0003, 5, 6, 8, 20, 10]]), 0, 0, 6, 4, 20)
result_q = result_assemble[0]
result_qd = result_assemble[1]
result_qdd = result_assemble[2]
time_vec = result_assemble[3]
matrix_eval = np.zeros((len(result_q), 3))
for i in range (len(result_q)):
    matrix_eval[i, 0] = result_q[i, 2]
    matrix_eval[i, 1] = result_qd[i, 2]
    matrix_eval[i, 2] = result_qdd[i, 2]

print(matrix_eval)

for i in range(3):
    x_point = matrix_eval[i, :]
    y_point = matrix_eval[i+1, :]
    #while n < t1 + delta/100: # plus a small margin to compensate floating point error
        #func[i] = hermite_basis(n)[0, :].dot(coeff)
        #time_spline[i] = n
        #print(f'array func at index {i} has the value of {func[i]}, n = {n}')
        #i = i + 1
        #print(i)
        #n = n + delta
    
    tx = quintic_hermite_interpolation(1, x_point[0], x_point[1], x_point[2], y_point[0], y_point[1], y_point[2])
    print(tx)

"""