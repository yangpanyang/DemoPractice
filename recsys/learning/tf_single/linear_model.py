#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/14 10:43 PM
    @Author : Caroline
    @File : 用tensorflow做简单线性回归的case
    @Description :
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    n_samples = 1000
    batch_size = 100
    # dataset
    X_data = np.arange(100, step=.1)
    y_data = X_data + 20 * np.sin(X_data/10)
    X_data = np.reshape(X_data, (n_samples,1))
    y_data = np.reshape(y_data, (n_samples,1))

    # create graph
    X = tf.placeholder(tf.float32, shape=(batch_size, 1))
    y = tf.placeholder(tf.float32, shape=(batch_size, 1))
    with tf.variable_scope("linear-regression"): #, reuse=True):
        W = tf.get_variable("weights", (1, 1), initializer=tf.random_normal_initializer())
        b = tf.get_variable("bias", (1,), initializer=tf.constant_initializer(0.0))
        y_pred = tf.matmul(X, W) + b
        loss = tf.reduce_sum((y - y_pred)**2/n_samples)
        opt = tf.train.AdamOptimizer() # create an optimizer
        opt_operation = opt.minimize(loss)
        # compute
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for i in range(5000):
                indices = np.random.choice(n_samples, batch_size)
                X_batch, y_batch = X_data[indices], y_data[indices]
                t, loss_val = sess.run([opt_operation, loss], feed_dict={X: X_batch, y: y_batch})
            print('iter: ', i, 'loss: ', loss_val, 'W: ', W.eval(), 'b: ', sess.run(b)) # 'y_pred: ', sess.run(y_pred, feed_dict={X: X_batch})
            # plt.scatter(X_batch, y_batch)
            # plt.scatter(X_batch, sess.run(y_pred, feed_dict={X: X_batch}))