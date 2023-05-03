import roboticstoolbox as rtb
import numpy as np
import math
from math import pi
from roboticstoolbox import tools as tools
import spatialmath as sm
import spatialgeometry as sg
import matplotlib as mpl
import matplotlib.pyplot as plt
from call_tau import *
from scipy.stats import truncnorm
from scipy.integrate import simpson
from plot_traj import *
from traj import *
from HKA_kalman_gain import *
from energy_est import *
import time as ti

def random_traj(vel_max, t_accel_rand, q0, qf, tf, time):
    q = []
    qd = []
    qdd = []
    a_accel_rand = (vel_max / t_accel_rand)
    t_brake_rand = 2 * (qf - q0) / vel_max + t_accel_rand - tf
    a_brake_rand = vel_max / (tf - t_brake_rand)
    for tk in time:
        if tk < 0:
            qk = q0
            qdk = 0
            qddk = 0
        elif tk <= t_accel_rand:
            qk = q0 + 0.5 * a_accel_rand * tk**2
            qdk = a_accel_rand * tk
            qddk = a_accel_rand
        elif tk <= t_brake_rand:
            qk = q0 + 0.5 * a_accel_rand * t_accel_rand**2 + vel_max * (tk - t_accel_rand)
            qk = vel_max * tk + q0 + 0.5 * a_accel_rand * t_accel_rand**2 - vel_max * t_accel_rand
            qdk = vel_max
            qddk = 0
        elif tk <= tf:
            qk = q0 + 0.5 * a_accel_rand * t_accel_rand**2 + vel_max * (t_brake_rand - t_accel_rand) + (vel_max * (tk - t_brake_rand) - 0.5 * a_brake_rand * (tk - t_brake_rand)**2)
            qdk = vel_max - a_brake_rand * (tk - t_brake_rand)
            qddk = -a_brake_rand
        else:
            qk = qf
            qdk = 0
            qddk = 0
        q.append(qk)
        qd.append(qdk)
        qdd.append(qddk)

    return (np.array(q), np.array(qd), np.array(qdd))


def hka_profile_opt(N, Nbest, traj_time, step):
    start_time = ti.time()
    vel_max = np.zeros(6)
    a = np.zeros(6)
    t_accel = np.zeros(6)
    t_brake = np.zeros(6)
    t_min = np.zeros(6)
    t_max = np.zeros(6)
    traj = generate_traj_time(traj_time, step)
    plot_trajectory(traj)
    time = traj[6]
    angle = np.zeros((6, step))
    velocity = np.zeros((6, step))
    accel = np.zeros((6, step))
    t0 = 0 # start
    tf = time[-1] # finish
    mu_t = np.zeros(6)
    sig_t =  np.zeros(6)
    t_accel_rand = np.zeros((6, N))
    q0 = np.zeros(6)
    qf = np.zeros(6)
    q_mat = np.zeros((N, 6, step))
    qd_mat = np.zeros((N, 6, step))
    qdd_mat = np.zeros((N, 6, step))
    iter = 0
    max_iter = 150
    post_t_rand = np.zeros((6, Nbest))

    for j in range(6):

        angle[j, :] = traj[j].q
        velocity[j, :] = traj[j].qd
        accel[j, :] = traj[j].qdd
        q0[j] = angle[j, 0]
        qf[j] = angle[j, -1]
        vel_max[j] = (qf[j] - q0[j]) / tf * 1.5
        t_accel[j] = np.round((q0[j] - qf[j] + vel_max[j] * tf) / vel_max[j], 2) # end of acceleration, rounded to 2 decimals to exactly match the time points in traj[6]
        t_brake[j] = np.round(tf - t_accel[j], 2) # start of braking
        #i_accel = np.where(time == t_accel)[0][0]
        #i_brake = np.where(time == t_brake)[0][0]
        a[j] = vel_max[j] / t_accel[j]
        t_min[j] = abs(vel_max[j] / 150)
        t_max[j] = abs(tf / 3 + vel_max[j] / 150)

    mu_t = t_accel # initialize mean vector
    sig_t = (t_max - t_min) / 2 # initialize std.dev. vector
    q_torq_og = np.transpose(angle)
    qd_torq_og = np.transpose(velocity)
    qdd_torq_og = np.transpose(accel)
    en_og = calculate_energy(q_torq_og, qd_torq_og, qdd_torq_og, time)
    print(f'original total energy consumption: {en_og} J')

    # HKA starts here
    while iter <= 150:


        for j in range(6):
            lb = (t_min[j] - mu_t[j]) / sig_t[j]
            ub = (t_max[j] - mu_t[j]) / sig_t[j]
            trunc_gen_t = truncnorm(lb, ub, loc=mu_t[j], scale=sig_t[j])
            t_accel_rand[j, :] = trunc_gen_t.rvs(size=N)


    #for i in range(50):
    #fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained')
        energy_list_def = [('Energy','f8'), ('Number','i2')] # list that contains the energy consumption and the corresponding row index from assble_qdd
        energy_list = np.zeros((N), dtype = energy_list_def)

        for i in range(N):
            power_val = []
            for j in range(6):
                max_velocity = vel_max[j]
                t_accel_r = t_accel_rand[j, i]
                q0_r = q0[j]
                qf_r = qf[j] 
                tg = random_traj(max_velocity, t_accel_r, q0_r, qf_r, tf, time)
                q_mat[i, j, :] = tg[0]
                qd_mat[i, j, :] = tg[1]
                qdd_mat[i, j, :] = tg[2]

            """ to plot search domain
            fig.suptitle(f'Search domain for Joint 1', fontsize=16)
            ax1.plot(time, q_mat[i, 0, :])
            ax1.set_xlabel('Travel Time in s')
            ax1.set_ylabel('Joint angle in rad')
            ax2.plot(time, qd_mat[i, 0, :])
            ax2.set_xlabel('Travel Time in s')
            ax2.set_ylabel('Joint velocity in rad/s')
            """


            q_torq = np.transpose(q_mat[i, :, :])
            qd_torq = np.transpose(qd_mat[i, :, :])
            qdd_torq = np.transpose(qdd_mat[i, :, :])
            energy_list[i] = (calculate_energy(q_torq, qd_torq, qdd_torq, time), i)
            #for k in range(len(q_torq)):
            #    torq_vec = cal_tau(q_torq[k, :], qd_torq[k, :], qdd_torq[k, :])
            #    vel_vec = qd_torq[k, :]
            #    power_vec = np.linalg.norm(np.multiply(torq_vec, vel_vec), 1)
            #    power_val.append(power_vec)

            #energy_list[i] = (simpson(power_val, time), i)
        sorted_energy_list = np.sort(energy_list, order='Energy')
        #print(sorted_energy_list)
        num_array = sorted_energy_list['Number']
        t_rand_index = num_array[0 : Nbest]
        #print(t_rand_index)
        #print(t_accel_rand)
        for j in range(6):
            post_t_rand[j, :] = [t_accel_rand[j, i] for i in t_rand_index]
        #print(post_t_rand)
        mu_t_rand = np.mean(post_t_rand, 1)
        var_t_rand = np.var(post_t_rand, 1)
        new_mu_sig_t = kalman_gain(sig_t, var_t_rand, mu_t, mu_t_rand)
        mu_t = new_mu_sig_t[0]
        sig_t = new_mu_sig_t[1]
        print(var_t_rand)

        if all(i < 1e-6 for i in var_t_rand) == True:
            print(f'exited HKA at iter = {iter}')
            break

        iter = iter + 1

    energy_opt = energy_list[num_array[0]]
    result_q = q_mat[num_array[0], :, :]
    result_qd = qd_mat[num_array[0], :, :]
    result_qdd = qdd_mat[num_array[0], :, :]
    np.savetxt('prof_result_q.txt', result_q)
    np.savetxt('prof_result_qd.txt', result_qd)
    np.savetxt('prof_result_qdd.txt', result_qdd)
    print(f'optimized total energy consumption: {energy_opt} J')
    print(f'Optimization runtime: {ti.time() - start_time} seconds')

hka_profile_opt(50, 5, 2, 201)

decision = 1
if decision == 1:
    for j in range(6):
        result_q = np.loadtxt('prof_result_q.txt')
        result_qd = np.loadtxt('prof_result_qd.txt')
        result_qdd = np.loadtxt('prof_result_qdd.txt')
        joint_data = generate_traj_time(2, 201)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
        ax1.plot(joint_data[6], result_q[j, :], label=f'optimized traj for joint {j+1}')
        ax1.plot(joint_data[6], joint_data[j].q, label=f'orignal traj for joint {j+1}')
        ax1.set_xlabel('Travel time in s')
        ax1.set_ylabel('Joint angle in rad')
        ax1.legend()
        ax2.plot(joint_data[6], result_qd[j, :], label=f'optimized traj for joint {j+1}')
        ax2.plot(joint_data[6], joint_data[j].qd, label=f'orignal traj for joint {j+1}')
        ax2.set_xlabel('Travel time in s')
        ax2.set_ylabel('Joint velocity in rad/s')
        ax3.plot(joint_data[6], result_qdd[j, :], label=f'orignal traj for joint {j+1}')
        ax3.plot(joint_data[6], joint_data[j].qdd, label=f'orignal traj for joint {j+1}')
        ax3.set_xlabel('Travel time in s')
        ax3.set_ylabel('Joint acceleration in rad/s^2')
        fig.suptitle(f"Trajectory for Joint {j+1}", fontsize=16)
        plt.show()

#plt.show()
"""      
for tk in t:
    if tk < 0:
        pk = q0
        pdk = 0
        pddk = 0
    elif tk <= tb:
        # initial blend
        pk = q0 + a / 2 * tk**2
        pdk = a * tk
        pddk = a
    elif tk <= (tf - tb):
        # linear motion
        pk = (qf + q0 - V * tf) / 2 + V * tk
        pdk = V
        pddk = 0
    elif tk <= tf:
        # final blend
        pk = qf - a / 2 * tf**2 + a * tf * tk - a / 2 * tk**2
        pdk = a * tf - a * tk
        pddk = -a
    else:
        pk = qf
        pdk = 0
        pddk = 0
    p.append(pk)
    pd.append(pdk)
    pdd.append(pddk)
    """
# The following section is for graph generation. Uncomment to visualise q, qd and qdd
"""
rdr = generate_traj_time(2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(rdr[6], np.round(rdr[0].q, 6), label='q1')
ax1.plot(rdr[6], np.round(rdr[1].q, 6), label='q2')
ax1.plot(rdr[6], np.round(rdr[2].q, 6), label='q3')
ax1.plot(rdr[6], np.round(rdr[3].q, 6), label='q4')
ax1.plot(rdr[6], np.round(rdr[4].q, 6), label='q5')
ax1.plot(rdr[6], np.round(rdr[5].q, 6), label='q6')
ax1.set_xlabel('Travel Time in s')
ax1.set_ylabel('joint angle in rad')
ax1.legend()

ax2.plot(rdr[6], np.round(rdr[0].qd, 6), label='q1')
ax2.plot(rdr[6], np.round(rdr[1].qd, 6), label='q2')
ax2.plot(rdr[6], np.round(rdr[2].qd, 6), label='q3')
ax2.plot(rdr[6], np.round(rdr[3].qd, 6), label='q4')
ax2.plot(rdr[6], np.round(rdr[4].qd, 6), label='q5')
ax2.plot(rdr[6], np.round(rdr[5].qd, 6), label='q6')
ax2.set_xlabel('Travel Time in s')
ax2.set_ylabel('joint velocity in 1/s')
ax2.legend()

ax3.plot(rdr[6], np.round(rdr[0].qdd, 6), label='q1')
ax3.plot(rdr[6], np.round(rdr[1].qdd, 6), label='q2')
ax3.plot(rdr[6], np.round(rdr[2].qdd, 6), label='q3')
#ax3.plot(rdr[6][0], rdr[2].qdd[0], 'o')
ax3.plot(rdr[6], np.round(rdr[3].qdd, 6), label='q4')
ax3.plot(rdr[6], np.round(rdr[4].qdd, 6), label='q5')
ax3.plot(rdr[6], np.round(rdr[5].qdd, 6), label='q6')
ax3.set_xlabel('Travel Time in s')
ax3.set_ylabel('joint acceleration in $1/s^2$')
ax3.set_ylim(bottom=-5, top=5)
ax3.legend()

plt.show()

#fig1 = plt.figure()
#fig1 = plt.plot(rdr[6], np.round(rdr[0].q, 6))
#plt.xlabel('Trajectory Time in s')
#plt.ylabel('Joint angle in rad')
#plt.show()
"""