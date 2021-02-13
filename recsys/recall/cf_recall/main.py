#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/5 8:12 PM
    @Author : Caroline
    @File : 函数入口
    @Description : 测试各种cf算法的实现
"""

from experiment import Experiment

if __name__ == '__main__':
    print('Test CF algorithm.')
    # UserIIF
    M, N = 8, 10
    for K in [5, 10, 20, 40, 80, 160]:
        cf_exp = Experiment(M, K, N)
        cf_exp.run()
    # ItemIUF
    M, N = 8, 10
    for K in [5, 10, 20, 40, 80, 160]:
        cf_exp = Experiment(M, K, N, rt='ItemIUF')
        cf_exp.run()
    # ItemCF_Norm
    M, N = 8, 10
    for K in [5, 10, 20, 40, 80, 160]:
        cf_exp = Experiment(M, K, N, rt='ItemCF_Norm')
        cf_exp.run()
    # LFM
    M, N = 8, 10
    for r in [1, 2, 3, 5, 10, 20]:
        exp = Experiment(M=M, N=N, ratio=r, step=10, rt='LFM')
        exp.run()