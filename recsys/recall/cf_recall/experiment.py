#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/5 7:58 PM
    @Author : Caroline
    @File : 实验测试类
    @Description : 便于测试各种CF算法
"""

import os
from dataset import timmer
from dataset import Dataset
from cf_algo import *
from metric import Metric


class Experiment():
    def __init__(self, M=5, K=8, N=10, \
                 K_latent=100, ratio=1, \
                 lr=0.02, step=100, lmbda=0.01, \
                 fp=os.path.join('../../data/ml-1m', 'ratings.dat'), \
                 rt='UserIIF'):
        '''
        :params: M, 进行多少次实验, 类似M折交叉验证
        :params: K, TopK相似用户的个数
        :params: N, TopN推荐物品的个数
        :params: K_latent, 隐语义个数
        :params: ratio, 正负样本比例
        :params: lr, 学习率
        :params: step, 训练步长
        :params: lmbda, 正则化系数
        :params: fp, 数据文件路径
        :params: rt, 推荐算法类型
        '''
        print('Building data with function: {}'.format(rt))
        self.M = M
        self.K = K
        self.N = N
        self.K_latent = K_latent
        self.ratio = ratio
        self.lr = lr
        self.step = step
        self.lmbda = lmbda
        self.fp = fp
        self.rt = rt
        self.alg = { \
            'UserIIF': UserIIF, \
            'ItemIUF': ItemIUF, \
            'ItemCF_Norm': ItemCF_Norm, \
            'LFM': LFM \
            }

    # 定义单次实验
    @timmer
    def worker(self, train, test):
        '''
        :parmas: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        print('Starting train data with function: {}'.format(self.rt))
        if self.rt == 'LFM':
            getRecommendation, P, Q = self.alg[self.rt](train, self.ratio, self.K_latent, \
                                                        self.lr, self.step, self.lmbda, self.N)
        else:
            getRecommendation = self.alg[self.rt](train, self.K, self.N)  # train
        print('Starting cacluate evaluation metrics with function: {}'.format(self.rt))
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均值
    @timmer
    def run(self):
        metrics = { \
            'Precision': 0, \
            'Recall': 0, \
            'Coverage': 0, \
            'Popularity': 0 \
            }
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test = dataset.splitData(self.M, ii)
            print('Experiment {}: train data users = {}, test data users = {}'.format( \
                ii, len(train), len(test)))
            metric = self.worker(train, test)
            metrics = {k: (metrics[k] + metric[k]) for k in metrics}
        metrics = {k: (metrics[k] / self.M) for k in metrics}
        print('Average Result (M={}, K={}, N={}): {}'.format( \
            self.M, self.K, self.N, metrics))
