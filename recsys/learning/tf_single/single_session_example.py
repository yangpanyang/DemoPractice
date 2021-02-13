#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/9 11:50 AM
    @Author : Caroline
    @File : tensorflow简单session实现
    @Description : 使用tensorflow搭建简单网络结构，并绘图至tensorboard展示
"""

import tensorflow as tf
# from tensorflow.keras import layers  # 1.15版本
from tensorflow import feature_column
# from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.keras import backend as K

columns = [
    'id',
    'click',
    'hour',
    'C1',
    'banner_pos',
    'site_id',
    'site_domain',
    'site_category',
    'app_id',
    'app_domain',
    'app_category',
    'device_id',
    'device_ip',
    'device_model',
    'device_type',
    'device_conn_type',
    'C14',
    'C15',
    'C16',
    'C17',
    'C18',
    'C19',
    'C20',
    'C21',
]


# Option I, read data into memory: pandas dataframe
# Option II, read data by batch
def get_dataset():
    # batch_size：把文件切成一个个的batch去读
    # column_names：指明数据的schema
    # num_epochs：表示文件数据过几圈
    # label_name：声明label是schema中的哪一列
    dataset = tf.data.experimental.make_csv_dataset("../data/avazu-ctr-prediction/train",
                                                    batch_size=2,
                                                    column_names=columns,
                                                    label_name='click',
                                                    num_epochs=1)
    return dataset


# trans sparse space to embedding
def fc_transform(feature_name, hash_bucket_size, dtype=tf.string):
    f = feature_column.categorical_column_with_hash_bucket(
        feature_name, hash_bucket_size=hash_bucket_size, dtype=dtype)
    f1 = feature_column.embedding_column(f, 4)  # 可以被训练的，就需要在main里面对变量进行初始化
    # 把feature_column对象转为 dense tensor，所有的feature_column都必须做这步
    feature_layer = tf.keras.layers.DenseFeatures([f1])  # 继承自Layer基类，会自动实现call方法
    return feature_layer

class SafeKEmbedding(tf.keras.layers.Embedding):
    def compute_mask(self, inputs, mask=None):
        oov_mask = tf.less(inputs, self.input_dim)
        if not self.mask_zero:
            return oov_mask
        return tf.logical_and(oov_mask, tf.not_equal(inputs, 0))

    def call(self, inputs):
        out = super().call(inputs)
        mask = tf.expand_dims(tf.cast(self.compute_mask(inputs), K.dtype(out)), -1)
        return out * mask

def plot_structual():
    # Step I, read data
    raw_train_data = get_dataset()

    # Step II, consume data
    # 建立了一个iterator，可以对里面的数据使用next方法进行遍历
    k = raw_train_data.make_initializable_iterator()  # iterator需要自己初始化，在main里面实现了
    features, labels = k.get_next()  # feature是OrderedDict，label是普通的tensor

    # Step III, use feature column to do feature transformation
    device_ip = fc_transform('device_ip', 100)(features)
    C1 = fc_transform('C1', 100, dtype=tf.int32)(features)
    ip = features['C1']
    emb_ip = SafeKEmbedding(100, 6, mask_zero=True, name='embedding_device_ip')(features['C1'])

    # step IV, network body
    with tf.variable_scope("haha"):
        t1 = tf.keras.layers.Concatenate(axis=-1)([device_ip, C1])
        t2 = tf.keras.layers.Dense(4, activation='relu')(t1)
        t3 = tf.keras.layers.Dense(1)(t2)
    # print(t1)

    # Step V, loss and optimizer
    # squeeze把2维变成1维
    # 输出：得到一个batch size大小的loss
    loss_op = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(tf.cast(labels, tf.float32), [-1, 1]),
                                                      logits=tf.squeeze(t3))  # 直接计算交叉熵，这样计算精度相对好一点
    update_op = tf.train.AdamOptimizer().minimize(loss_op)
    # update_op = tf.train.AdamOptimizer().compute_gradients(loss_op)
    # tf.clip_by_norm(update_op)  # 做梯度裁剪，避免出现梯度爆炸
    # update_op = tf.train.AdamOptimizer().apply_gradients(update_op)
    # print('loss: ', loss_op, update_op)
    return k, loss_op, update_op, ip, emb_ip


def plot_board(loss_op):
    # Step VI, tensorboard
    # 声明summary的name、对应的tensor
    tf.summary.scalar('loss', tf.reduce_mean(loss_op))  # 每个batch_size里的loss求mean
    # 把之前所有的summary都merge到一起
    merged_op = tf.summary.merge_all()  # 写tensorboard的op
    eval_writer = tf.summary.FileWriter(
        '../model_dir/single_session')  # 写tensorboard文件的位置
    return merged_op, eval_writer


# Step VII, run with session
if __name__ == '__main__':
    k, loss_op, update_op, ip, emb_ip = plot_structual()
    merged_op, eval_writer = plot_board(loss_op)
    with tf.Session() as sess:  # 构建session
        sess.run(k.initializer)  # 初始化iterator，对数据做了初始化
        sess.run(tf.global_variables_initializer())  # 初始化全局可训练的变量

        print(sess.run(ip))
        print(sess.run(emb_ip))

        # 多轮训练
        for i in range(1000):
            if i % 10 == 0:  # evaluation部分，当前数据不进行训练
                merged, loss_value = sess.run(
                    [merged_op, tf.reduce_mean(loss_op)])  # tf.reduce_sum(loss_op) 计算每个batch size的loss
                print(i, merged, loss_value)
                eval_writer.add_summary(merged, i)
            else:  # training
                sess.run(update_op)
