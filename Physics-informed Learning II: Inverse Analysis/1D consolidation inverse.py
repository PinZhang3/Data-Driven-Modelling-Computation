# ----------------------------------------------------------------------
# Project Name: Data-Driven Modelling & Computation
# Description : This code is for 1D consolidation inverse analysis
# Author      : Pin ZHANG, National University of Singapore
# Contact     : pinzhang@nus.edu.sg
# Created On  : 26 Jun 2025
# Repository  : https://github.com/PinZhang3/Data-Driven-Modelling-and-Computation
# ----------------------------------------------------------------------
# Notes:
# This library is under active development. Contributions are welcome!
# Copyright belongs to Pin ZHANG and use of this code for commercial applications or
# profit-driven ventures requires explicit permission from the author(s)
# ----------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata

class PiNet:
    def __init__(self, CP, Ini, Top_t, Bot_t, X_f, layers, lb, ub):
        
        # domain Boundary
        self.lb = lb
        self.ub = ub
        
        # Init for Solution
        self.t_cp = CP[:, 0:1]
        self.x_cp = CP[:, 1:2]
        self.u_cp = CP[:, 2:3]

        self.t0 = Ini[:, 0:1]
        self.x0 = Ini[:, 1:2]
        self.u0 = Ini[:, 2:3]

        self.t_lb = Bot_t[:, 0:1]
        self.x_lb = Bot_t[:, 1:2]

        self.t_ub = Top_t[:, 0:1]
        self.x_ub = Top_t[:, 1:2]

        self.t_f = X_f[:, 0:1]
        self.x_f = X_f[:, 1:2]

        # tf placeholders
        self.t_cp_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_cp_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_cp_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, 1])

        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # initialize weights and biases, and initial guess
        self.Cv = tf.Variable([1.0], dtype=tf.float32)  # initial trial
        self.layers = layers
        self.weights, self.biases = self.initialize_nn(layers)

        # loss function
        self.u0_pred, _, _ = self.net_u(self.t0_tf, self.x0_tf)
        _, self.u_x_lb_pred, _ = self.net_u(self.t_lb_tf, self.x_lb_tf)
        self.u_ub_pred, _, _ = self.net_u(self.t_ub_tf, self.x_ub_tf)
        _, _, self.f_pred = self.net_u(self.t_f_tf, self.x_f_tf)
        self.u_cp_pred, _, _ = self.net_u(self.t_cp_tf, self.x_cp_tf)

        self.loss = tf.reduce_sum(tf.square(self.u0_tf - self.u0_pred)) + \
                        tf.reduce_sum(tf.square(self.u_cp_pred - self.u_cp)) + \
                        tf.reduce_sum(tf.square(self.f_pred))

        # Optimizer
        # self.optimizer_LBFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
        #                                                             method='L-BFGS-B',
        #                                                             options={'maxiter': 10000,
        #                                                                      'maxfun': 10000,
        #                                                                      'maxcor': 100,
        #                                                                      'maxls': 100,
        #                                                                      'gtol': 1e-03})

        self.optimizer_Adam = tf.train.AdamOptimizer().minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_nn(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32),
                           dtype=tf.float32)

    def neural_network(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_u(self, t, x):
        Cv = self.Cv
        X = tf.concat([t,x],1)
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        u = self.neural_network(H, self.weights, self.biases)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t - Cv / 365 / 24 * u_xx
        return u, u_x, f

    def callback(self, loss, Cv):
        # print('Loss: %e, Cv: %.10f' % (loss, Cv))
        self.loss_history.append(loss)
        self.Cv_history.append(Cv)
        
    def train(self, N_iter):
        tf_dict = {self.t0_tf: self.t0, self.x0_tf: self.x0,
                   self.u0_tf: self.u0,
                   self.t_lb_tf: self.t_lb, self.x_lb_tf: self.x_lb,
                   self.t_ub_tf: self.t_ub, self.x_ub_tf: self.x_ub,
                   self.t_f_tf: self.t_f, self.x_f_tf: self.x_f,
                   self.t_cp_tf: self.t_cp, self.x_cp_tf: self.x_cp, self.u_cp_tf: self.u_cp}

        self.loss_history = []
        self.Cv_history = []
        self.Cv_history.append(self.sess.run(self.Cv, tf_dict)) # initial value
        for it in range(N_iter):
            self.sess.run(self.optimizer_Adam, tf_dict)
            loss = self.sess.run(self.loss, tf_dict)
            Cv = self.sess.run(self.Cv, tf_dict)
            self.loss_history.append(loss)
            self.Cv_history.append(Cv)

            if it % 1000 == 0:
                print('It: %d, Loss: %.3e' % (it, loss))
                
        # self.optimizer_LBFGS.minimize(self.sess,
        #                             feed_dict = tf_dict,
        #                             fetches = [self.loss, self.Cv],
        #                             loss_callback = self.callback)
    
    def predict(self, t_star, x_star):
        
        u_star = self.sess.run(self.u0_pred, {self.t0_tf: t_star, self.x0_tf: x_star})  
        f_star = self.sess.run(self.f_pred, {self.t_f_tf: t_star, self.x_f_tf: x_star})
               
        return u_star, f_star    

if __name__ == "__main__": 

    # doman bounds
    lb = np.array([0.0, -0.03])
    ub = np.array([4.0, 0])
    
    # load data

    Exact = scipy.io.loadmat('consol_1D_sub.mat')

    t_all = Exact['tspan'].flatten()[:, None] # T*1
    x_all = Exact['z'].flatten()[:, None]     # N*1
    Exact_sol_all = np.real(Exact['u_sub'])   # N*T

    ###########################################################################
    # for investigating data dependency
    interval_x = 1
    interval_t = 1
    N_t = int((len(t_all) - 1) / interval_t)
    N_x = int((len(x_all) - 1) / interval_x)

    t_sol = np.zeros((N_t + 1, 1))
    x_sol = np.zeros((N_x + 1, 1))
    Exact_sol = np.zeros((N_x+1, N_t + 1))

    for i in range(N_t + 1):
        if i != N_t + 1:
            t_sol[i] = t_all[interval_t * i]
        else:
            t_sol[i] = t_all[-1]

    for i in range(N_x + 1):
        if i != N_x + 1:
            x_sol[i] = x_all[interval_x * i]
        else:
            x_sol[i] = x_all[-1]

    for i in range(N_x + 1):
        if i != N_x + 1:
            for j in range(N_t + 1):
                if j != N_t + 1:
                    Exact_sol[i, j] = Exact_sol_all[interval_x * i, interval_t * j]
                else:
                    Exact_sol[i, j] = Exact_sol_all[interval_x * i, -1]
        else:
            for j in range(N_t + 1):
                Exact_sol[i, j] = Exact_sol_all[-1, interval_t * j]

    ###########################################################################

    T_sol, X_sol = np.meshgrid(t_sol, x_sol)
    
    t_sol_star = T_sol.flatten()[:, None]
    x_sol_star = X_sol.flatten()[:, None]
    X_sol_star = np.hstack((t_sol_star, x_sol_star))

    u_sol_star = Exact_sol.flatten()[:, None]
     
    # collocation points -- known values
    CP = np.hstack((t_sol_star, x_sol_star, u_sol_star))

    # initial and boundary conditions
    T = t_sol.shape[0]
    N = x_sol.shape[0]

    Ini = np.zeros((N, 3))
    Ini[:, 0:1] = np.zeros((N, 1))
    Ini[:, 1:2] = x_sol
    Ini[:, 2:3] = Exact_sol[:, 0:1]

    Top_t = np.zeros((T, 2))
    Top_t[:, 0:1] = t_sol
    Top_t[:, 1:2] = ub[1] * np.ones((T, 1))

    Bot_t = np.zeros((T, 2))
    Bot_t[:, 0:1] = t_sol
    Bot_t[:, 1:2] = lb[1] * np.ones((T, 1))

    # PDE residuals
    X_f_train = np.hstack((t_sol_star, x_sol_star))

    layers = [2, 64, 64, 64, 64, 1]
    
    # Model
    model = PiNet(CP, Ini, Top_t, Bot_t, X_f_train, layers, lb, ub)
    N_adam = 6000
    model.train(N_iter=N_adam)
    u_pred, f_pred = model.predict(t_sol_star, x_sol_star)

    error_u = np.linalg.norm(u_sol_star-u_pred,2)/np.linalg.norm(u_sol_star,2)

    print('Error u: %e' % (error_u))

    U_pred = griddata(X_sol_star, u_pred.flatten(), (T_sol, X_sol), method='cubic')
    F_pred = griddata(X_sol_star, f_pred.flatten(), (T_sol, X_sol), method='cubic')
    U_sol_star = griddata(X_sol_star, u_sol_star.flatten(), (T_sol, X_sol), method='cubic')

    ############################# Plotting ###############################

    rc = {"font.family": "serif", "mathtext.fontset": "stix", "font.size": 16}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"]

    loss = model.loss_history
    fig = plt.figure(1, figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(loss)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='both', direction='in', which='both')

    Cv = model.Cv_history
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot([0, N_adam], [2.3, 2.3], 'r--')
    ax2.plot(Cv)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cv')
    ax2.tick_params(axis='both', direction='in', which='both')
    plt.show()

    # np.savetxt('1D_U_actu_sub_inverse.csv', U_sol_star, fmt='%0.10f', delimiter=',')
    # np.savetxt('1D_U_pred_sub_inverse.csv', U_pred, fmt='%0.10f', delimiter=',')
    # np.savetxt('1D_F_actu_sub_inverse.csv', F_pred, fmt='%0.10f', delimiter=',')