#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/5 8:29 PM
    @Author : Caroline
    @File : SVD算法实现
    @Description : 用矩阵实现SVD两个算法
"""

import numpy as np


class SVD:
    def __init__(self, learning_rate, regularized_rate, max_step, n_users, n_items, n_factors):
        self.learning_rate = learning_rate
        self.regularized_rate = regularized_rate
        self.max_step = max_step
        self.bu = np.zeros(n_users, np.double)
        self.bi = np.zeros(n_items, np.double)
        self.pu = np.zeros((n_users, n_factors), np.double)
        self.qi = np.zeros((n_items, n_factors), np.double)
        self.mean = 0

    def get_pred_value(self, u, i):
        return self.mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u], self.qi[i])

    # There is a problem here, can anyone find it?
    def fit(self, X, y):
        for index, row in X.iterrows():
            u, i, r = row['user_id'], row['item_id'], row['rating']
            err = r - self.get_pred_value(u, i)
            self.bu[u] += self.learning_rate * (err - self.regularized_rate * self.bu[u])
            self.bi[i] += self.learning_rate * (err - self.regularized_rate * self.bi[i])
            tmp = self.pu[u]
            self.pu[u] += self.learning_rate * (err * self.qi[i] - self.regularized_rate * self.pu[u])
            self.qi[i] += self.learning_rate * (err * tmp - self.regularized_rate * self.qi[i])
            if index == self.max_step:
                break
        return self

    # What if the one to predict is not inside our knowledge?
    def transform(self, X):
        result = [0] * len(X)
        for index, row in X.iterrows():
            u, i, r = row['user_id'], row['item_id'], row['rating']
            result[index] = self.get_pred_value(u, i)
        return result


class SVDPP:
    def __init__(self, learning_rate, regularized_rate, max_step, n_users, n_items, n_factors, ur_dict):
        self.learning_rate = learning_rate
        self.regularized_rate = regularized_rate
        self.max_step = max_step
        self.bu = np.zeros(n_users, np.double)
        self.bi = np.zeros(n_items, np.double)
        self.pu = np.zeros((n_users, n_factors), np.double)
        self.qi = np.zeros((n_items, n_factors), np.double)
        self.yj = np.zeros((n_items, n_factors), np.double)
        self.ur_dict = ur_dict  # user对所有item的实际打分字典
        self.mean = 0

    def get_pred_value(self, u, i, u_impl_fdb):
        return self.mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u] + u_impl_fdb, self.qi[i])

    # suppose we can get a user's rating related to all items he rated
    def get_user_rating(self, u):
        Iu = self.ur_dict[u].keys()  # [j for (j, _) in self.ur_dict[u]]
        sqrt_Iu = np.sqrt(len(Iu))  # 计算归一化项，使所有值缩放到同一个维度下比较
        u_impl_fdb = np.zeros(self.n_factors, np.double)
        for j in Iu:
            u_impl_fdb += self.yj[j] / sqrt_Iu
        # u_impl_fdb = np.sum(self.yj[[Iu]], axis=0) / sqrt_Iu
        return u_impl_fdb

    def fit(self, X, y):
        for index, row in X.iterrows():
            u, i, r = row['user_id'], row['item_id'], row['rating']

            # suppose we can get a user's rating related to all items he rated
            Iu = self.ur_dict[u].keys()  # [j for (j, _) in self.ur_dict[u]]
            sqrt_Iu = np.sqrt(len(Iu))  # 计算归一化项，使所有值缩放到同一个维度下比较
            u_impl_fdb = self.get_user_rating(u)

            err = r - self.get_pred_value(u, i, u_impl_fdb)
            self.bu[u] += self.learning_rate * (err - self.regularized_rate * self.bu[u])
            self.bi[i] += self.learning_rate * (err - self.regularized_rate * self.bi[i])
            tmp = self.pu[u]
            self.pu[u] += self.learning_rate * (err * self.qi[i] - self.regularized_rate * self.pu[u])
            self.qi[i] += self.learning_rate * (err * (tmp + u_impl_fdb) - self.regularized_rate * self.qi[i])
            for j in Iu:  # 只要User见过该Item，其 Item的分值 都应该被更新
                self.yj[j] += self.learning_rate * (err * self.qi[i] / sqrt_Iu - self.regularized_rate * self.yj[j])

            if index == self.max_step:
                break
        return self

    def transform(self, X):
        result = [0] * len(X)
        for index, row in X.iterrows():
            u, i, r = row['user_id'], row['item_id'], row['rating']
            u_impl_fdb = self.get_user_rating(u)
            result[index] = self.get_pred_value(u, i, u_impl_fdb)
        return result