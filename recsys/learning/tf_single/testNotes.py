#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/12/14 5:46 PM
    @Author : Caroline
    @File : #TODO
    @Description :
"""

import tensorflow as tf

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


# Step I, read data
# Option I, read data into memory: pandas dataframe
# Option II, read data by batch, 每次都现场拉数据
def get_dataset():
    dataset = tf.data.experimental.make_csv_dataset("../data/avazu-ctr-prediction/train",
                                                    batch_size=8,
                                                    column_names=columns,
                                                    label_name='click',
                                                    num_epochs=1)
    # dataset = dataset.batch(128)  # build pattern 的模式，可以直接对初始值做改动
    return dataset


# Step II, consume data
raw_train_data = get_dataset()
# trans to iterator: next(), end
k = raw_train_data.make_initializable_iterator()
# 不直接调用next方法，调用这个数据结构自己的next方法get_next()
features, labels = k.get_next()

# Step III, use feature

if __name__ == '__main__':
    print('Hello, world!\n')
    with tf.Session() as sess:
        print(labels)

# python要加载自己的库、标准库(不在python本身默认加载的库中，称之为标准库)、第三方库
# from collections import defaultdict, OrderedDict  # 标准库
# b = defaultdict(lambda: 0, a)  # a是一个dict，lambda是默认值
# OrderedDict：python本身实现字典的时候，内部使用哈希表的数据结构，遍历是无序的，而OrderedDict就会强制让每次遍历得到一样的值
