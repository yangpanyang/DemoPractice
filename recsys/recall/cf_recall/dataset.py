#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/5 8:01 PM
    @Author : Caroline
    @File : 数据处理类
    @Description : 各种数据清洗的方法实现
"""

import random
import time


# 定义装饰器，监控运行时间；对其他函数进行计时功能的扩展
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('Func %s, run time: %s' % (func.__name__, stop_time - start_time))
        return res

    return wrapper


# 数据处理
class Dataset():

    def __init__(self, fp):
        self.data = self.loadData(fp)

    @timmer
    def loadData(self, fp):
        data = []
        for line in open(fp):
            data.append(tuple(map(int, line.strip().split('::')[:2])))
        return data

    @timmer
    def splitData(self, M, k, seed=1):
        '''
        :params: data, 加载的所有(user, item)数据条目
        :params: M, 划分的数目，最后需要取M折的平均
        :params: k, 本次是第几次划分，k~[0, M)
        :params: seed, random的种子数，对于不同的k应设置成一样的
        :return: train, test
        '''
        train, test = [], []
        random.seed(seed)
        for user, item in self.data:
            if random.randint(0, M - 1) == k:
                test.append((user, item))
            else:
                train.append((user, item))

        # 处理成字典的形式 {user:set(items)}
        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict:
                    data_dict[user] = set()
                data_dict[user].add(item)
            data_dict = {k: list(data_dict[k]) for k in data_dict}
            return data_dict

        return convert_dict(train), convert_dict(test)
