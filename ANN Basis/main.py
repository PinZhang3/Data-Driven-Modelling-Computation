# ----------------------------------------------------------------------
# Project Name: Data-Driven Modelling & Computation
# Description : A neural network code for modelling of soil compressibility
# Author      : Pin ZHANG, National University of Singapore
# Contact     : pinzhang@nus.edu.sg
# Created On  : 25 Jun 2025
# Repository  : https://github.com/PinZhang3
# ----------------------------------------------------------------------
# Notes:
# This library is under active development. Contributions are welcome!
# Copyright belongs to Pin ZHANG and use of this code for commercial applications or
# profit-driven ventures requires explicit permission from the author(s)
# ----------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

class NN:
    def __init__(self, layer, N_grad, N_input, N_output, lb, ub):

        self.layer = layer
        self.N_input = N_input
        self.N_output = N_output

        # mode
        self.num_grad = N_grad
        self.lb_input = lb[self.N_output:]
        self.ub_input = ub[self.N_output:]

        # Init for solution
        self.nn_init()

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # self.saver.save(self.sess, "NNi_wb/nn_wb.ckpt")
        # self.saver.restore(self.sess, "../NN_wb/nn_wb.ckpt")

    def initialize_NN(self, layers):
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

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def nn_init(self):
        self.weights, self.biases = self.initialize_NN(self.layer)
        self.saver = tf.train.Saver(var_list=[self.weights[l] for l in range(len(self.layer) - 1)]
                                             + [self.biases[l] for l in range(len(self.layer) - 1)])

        # tf placeholders for solution
        self.input_tf = tf.placeholder(tf.float32, shape=[None, self.N_input])
        self.output_tf = tf.placeholder(tf.float32, shape=[None, self.N_output])

        # prediction
        self.output_pred = self.net(self.input_tf)

        # loss
        self.loss = tf.reduce_mean(tf.square(self.output_tf - self.output_pred))

        # record loss
        self.loss_log = []

        # optimization
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, var_list=self.weights + self.biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 10000,
                                                                         'maxfun': 10000,
                                                                         'maxcor': 100,
                                                                         'maxls': 100,
                                                                         'gtol': 1e-04})

        self.optimizer_Adam = tf.train.AdamOptimizer()      # default learning rate = 0.001
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.weights + self.biases)

    def net(self, input_tf):
        H = 2.0 * (input_tf - self.lb_input) / (self.ub_input - self.lb_input) - 1.0
        X = H
        output_pred = self.neural_net(X, self.weights, self.biases)
        return output_pred

    def callback(self, loss_value):
        # print('loss_value: %e' % (loss_value))
        self.loss_log.append(loss_value)

    def train(self, N_epoch, input, output, batch):

        self.loss_history = []

        for it in range(N_epoch):
            # Randomly sample a mini-batch of indices
            idx = np.random.choice(input.shape[0], batch, replace=False)

            # Slice the input and output batches
            input_batch = input[idx]
            output_batch = output[idx]

            tf_dict = {self.input_tf: input_batch, self.output_tf: output_batch}
            self.sess.run(self.train_op_Adam, tf_dict)
            # if it % (self.num_grad/2) == 0:
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_log.append(loss_value)
            # print('It: %d, loss_value: %e' % (it, loss_value))

        # self.optimizer.minimize(self.sess,
        #                         feed_dict=tf_dict,
        #                         fetches=[self.loss],
        #                         loss_callback=self.callback)

    def predict(self, input):
        output_pred = self.sess.run(self.output_pred, {self.input_tf: input})
        return output_pred

###############################################################################
################################ Main Function ################################
###############################################################################

random.seed(2025)
data = pd.read_csv('Cc.csv').values
lb = np.min(data, axis=0)
ub = np.max(data, axis=0)

N_input = 3
N_out = 1
N_cut_off = int(0.8*len(data))
output = data[:, :N_out]
input = data[:, N_out:]
input_train = input[:N_cut_off, :]
output_train = output[:N_cut_off, :]
input_test = input[N_cut_off:, :]
output_test = output[N_cut_off:, :]

layer = [N_input, 32, N_out]
N_epoch = 1000  # Adam training epoch
N_grad = 0   # show distribution of gradients per N_grad
batch = 128
model = NN(layer, N_grad, N_input, N_out, lb, ub)
model.train(N_epoch, input_train, output_train, batch)
Cc_pred_train = model.predict(input_train)
Cc_pred_test = model.predict(input_test)
np.savetxt('Cc_pred.csv', Cc_pred_test, fmt='%0.10f', delimiter=',')

rc = {"font.family": "serif", "mathtext.fontset": "stix", "font.size": 16}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"]

fig_1 = plt.figure(1, figsize=(10, 5))
ax1 = fig_1.add_subplot(1, 2, 1)
R2_training = r2_score(output_train, Cc_pred_train)
mae_training = mean_absolute_error(output_train, Cc_pred_train)
ax1.scatter(output_train, Cc_pred_train, alpha = 0.5, s=20,
            marker='o', edgecolors='k', facecolors='none', label='Training')
ax1.set_xlim([lb[0], ub[0]])
ax1.set_ylim([lb[0], ub[0]])
ax1.set_xlabel('Exact Cc')
ax1.set_ylabel('Predict Cc')
ax1.tick_params(direction='in')

ax2 = fig_1.add_subplot(1, 2, 2)
R2_testing = r2_score(output_test, Cc_pred_test)
mae_testing = mean_absolute_error(output_test, Cc_pred_test)
ax2.scatter(output_test, Cc_pred_test, alpha = 0.5, s=20,
            marker='o', edgecolors='k', facecolors='none', label='Testing')
ax2.set_xlim([lb[0], ub[0]])
ax2.set_ylim([lb[0], ub[0]])
ax2.set_xlabel('Exact Cc')
ax2.set_ylabel('Predict Cc')
ax2.tick_params(axis='both', direction='in')

loss = model.loss_log
fig_2 = plt.figure(2, figsize=(6, 5))
ax1 = fig_2.add_subplot(1, 1, 1)
ax1.plot(loss)
ax1.set_yscale('log')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.tick_params(axis='both', direction='in')
plt.show()