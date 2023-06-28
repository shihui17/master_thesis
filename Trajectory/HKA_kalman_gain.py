import numpy as np

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
    a = 0.8 * (s**2 / (s**2 + np.max(P))) # slowdown factor to slow down the convergence, coefficient (in this case 0.8) can vary (see HKA book)
    #print(P)
    #print(var_q_post)
    sig_new = sig + a * (np.sqrt(P) - sig) # set new std. deviation
    mu_new = mu_post # set new mean
    
    return mu_new, sig_new