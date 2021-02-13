#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/14 10:24 PM
    @Author : Caroline
    @File : tensorflow使用指南
    @Description : 一些基础语法
"""

import numpy as np
import tensorflow as tf
# tf_single.InteractiveSession()

if __name__ == '__main__':
    # 构图环节
    a1 = tf.zeros((2,2))
    print('get_shape: ', a1.get_shape())
    b1 = tf.ones((2,2))
    # 使用array初始化
    a2 = np.array([[1, 2], [3, 4]])
    a2 = tf.convert_to_tensor(a2, dtype='float32')
    b2 = np.array([[5, 6], [2, 1]])
    b2 = tf.convert_to_tensor(b2, dtype='float32')
    # constant初始化
    c = tf.constant(5.0)
    d = tf.constant(6.0)
    e = c * d
    cdplus = tf.add(c, d)
    cdmultiply = tf.multiply(c, d)
    # 变量初始化
    state = tf.Variable(7, name="counter") # state=0, tf_single.cast(state, tf_single.int32)
    new_value = tf.add(state, tf.constant(1)) # new_value=state+1
    update = tf.assign(state, new_value) # state=new_value
    # 变量初始化
    W = tf.Variable(tf.zeros((2,2)), name="weights")
    R= tf.Variable(tf.random_normal((2,2)), name="random_weights")

    # 计算环节，只要计算就需要session
    with tf.Session() as sess:
        print('reshape(1, 4): ', tf.reshape(a2, (1, 4)).eval())
        print('reshape(4, 1):\n', sess.run(tf.reshape(a2, (4, 1))))
        # 求和、求积
        tf_sum = tf.reduce_sum(a2, reduction_indices=1) # 逐行求和
        print('reduce_sum: ', sess.run(tf_sum))
        print('matmul:\n', tf.matmul(a2, b2).eval()) # 矩阵内积
        print('constant multiply: ', e.eval())
        result = sess.run([cdplus, cdmultiply])
        print('tf run in list: ', result)
        # 只要用到变量，就需要初始化
        sess.run(tf.global_variables_initializer())
        sess.run(update)
        print('assign value: ', sess.run(state))
        print('random_normal:\n', sess.run(R))


    # placeholder
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)
    with tf.Session() as sess:
        print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))


    # variable_scope
    with tf.variable_scope("foo"):
        with tf.variable_scope("bar"):
            v = tf.get_variable("v", [1])
    assert v.name == "foo/bar/v:0"

    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
        tf.get_variable_scope().reuse_variables()
        v1 = tf.get_variable("v", [1])
    assert v1 == v

    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
    with tf.variable_scope("foo", reuse=True):
        v1 = tf.get_variable("v", [1])
    assert v1 == v