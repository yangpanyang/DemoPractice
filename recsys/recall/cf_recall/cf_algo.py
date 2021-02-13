#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/5 7:50 PM
    @Author : Caroline
    @File : #TODO
    @Description :
        * UserIIF: 基于改进的用户余弦相似度的推荐
        * ItemIUF: 基于改进的物品余弦相似度的推荐
        * ItemCF_Norm: 基于改进版归一化的物品余弦相似度的推荐
        * LFM: 隐语义模型（矩阵分解）
        * 数据：MovieLens的ml-1m数据集，它包含了6040名用户对大约3900部电影的1000209条评分记录
"""

import numpy as np
import math
from tqdm import trange


# 基于改进的用户余弦相似度的推荐
def UserIIF(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数, 设置取TopK相似用户数目
    :params: N, 超参数, 设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 建立item->user的倒排索引
    item_users = {}
    for user, items in train.items():
        for item in items:
            if item not in item_users:
                item_users[item] = []
            item_users[item].append(user)
    # 计算两两用户间的相似度
    sim = {}
    num = {}
    for item, users in item_users.items():
        for u in users:
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for v in users:
                if u == v:
                    continue
                if v not in sim[u]:
                    sim[u][v] = 0
                # 相比于 UserCF，主要是改进了这里
                sim[u][v] += 1 / math.log(1 + len(users))
    # 计算用户相似度矩阵
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])
    # 按照相似度倒排
    # {1:{4:0.5,3:0.8,5:0.3}} ==> {1:[(3,0.8),(4,0.5),(5,0.3)]}
    sorted_user_sim = { \
        k: list( \
            sorted(v.items(), \
                   key=lambda x: x[1], \
                   reverse=True \
                   ) \
            ) \
        for k, v in sim.items() \
        }

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for u, score in sorted_user_sim[user][:K]:  # get TopK个相似用户u
            for item in train[u]:  # get相似用户u的浏览记录items
                if item in seen_items:  # 保留未浏览记录
                    continue
                if item not in items:
                    items[item] = 0
                items[item] += score  # 相似度分数分数累加，score=sim[user][u]
        recs = list( \
            sorted(items.items(), \
                   key=lambda x: x[1], \
                   reverse=True \
                   ) \
            )[:N]
        return recs

    return GetRecommendation


# 基于改进的物品余弦相似度的推荐
def ItemIUF(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数, 设置取TopK相似用户数目
    :params: N, 超参数, 设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算两两物品间的相似度
    sim = {}
    num = {}
    for user, items in train.items():
        for i in items:
            if i not in num:
                num[i] = 0
            num[i] += 1
            if i not in sim:
                sim[i] = {}
            for j in items:
                if i == j:
                    continue
                if j not in sim[i]:
                    sim[i][j] = 0
                # 相比于ItemCF，主要是改进了这里
                sim[i][j] += 1 / math.log(1 + len(items))
    # 计算物品相似度矩阵
    for i in sim:
        for j in sim[i]:
            sim[i][j] /= math.sqrt(num[i] * num[j])
    # 按照相似度倒排
    # {1:{4:0.5,3:0.8,5:0.3}} ==> {1:[(3,0.8),(4,0.5),(5,0.3)]}
    sorted_item_sim = { \
        k: list( \
            sorted(v.items(), \
                   key=lambda x: x[1], \
                   reverse=True \
                   ) \
            ) \
        for k, v in sim.items() \
        }

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for item in seen_items:  # get浏览记录
            for i, score in sorted_item_sim[item][:K]:  # 根据每条记录，get TopK个相似物品i
                if i in seen_items:
                    continue
                if i not in items:
                    items[i] = 0
                items[i] += score  # 相似度分数累加
        recs = list( \
            sorted(items.items(), \
                   key=lambda x: x[1], \
                   reverse=True \
                   ) \
            )[:N]
        return recs

    return GetRecommendation


# 基于改进版归一化的物品余弦相似度的推荐
def ItemCF_Norm(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数, 设置取TopK相似用户数目
    :params: N, 超参数, 设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算两两物品间的相似度
    sim = {}
    num = {}
    for user, items in train.items():
        for i in items:
            if i not in num:
                num[i] = 0
            num[i] += 1
            if i not in sim:
                sim[i] = {}
            for j in items:
                if i == j:
                    continue
                if j not in sim[i]:
                    sim[i][j] = 0
                # 相比于ItemCF，主要是改进了这里
                sim[i][j] += 1 / math.log(1 + len(items))
    # 计算物品相似度矩阵
    for i in sim:
        for j in sim[i]:
            sim[i][j] /= math.sqrt(num[i] * num[j])
    # *** 改进：对相似度矩阵，按行最大值，进行归一化 ***
    for i, items in sim.items():
        maxj = -10000
        for j, score in items.items():  # 逐行取max
            maxj = max(score, maxj)
        if maxj == 0:
            continue
        for j, score in items.items():
            sim[i][j] /= maxj
    # 按照相似度倒排
    # {1:{4:0.5,3:0.8,5:0.3}} ==> {1:[(3,0.8),(4,0.5),(5,0.3)]}
    sorted_item_sim = { \
        k: list( \
            sorted(v.items(), \
                   key=lambda x: x[1], \
                   reverse=True \
                   ) \
            ) \
        for k, v in sim.items() \
        }

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for item in seen_items:  # get浏览记录
            for i, score in sorted_item_sim[item][:K]:  # 根据每条记录，get TopK个相似物品i
                if i in seen_items:
                    continue
                if i not in items:
                    items[i] = 0
                items[i] += score  # 相似度分数累加
        recs = list( \
            sorted(items.items(), \
                   key=lambda x: x[1], \
                   reverse=True \
                   ) \
            )[:N]
        return recs

    return GetRecommendation


# 隐语义模型（矩阵分解）
def LFM(train, ratio, K, lr, step, lmbda, N):
    '''
    :params: train, 训练数据
    :params: ratio, 负采样的正负比例
    :params: K, 隐语义个数
    :params: lr, 初始学习率
    :params: step, 迭代次数
    :params: lmbda, 正则化系数
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''
    # 把所有item取出来，并计算物品流行度，用于采样
    all_items = {}
    for user, items in train.items():
        for item in items:
            if item not in all_items:
                all_items[item] = 0
            all_items[item] += 1
    all_items = list(all_items.items())
    items = [x[0] for x in all_items]
    popularitys = [x[1] for x in all_items]

    # 负采样函数(按照物品流行度采样)
    def nSample(data, ratio):
        new_data = {}
        # 正样本
        for user, items in data.items():
            if user not in new_data:
                new_data[user] = {}
            for item in items:
                new_data[user][item] = 1
        # 负样本：对样本中的每个user，采样ratio倍的负样本
        for user in new_data:
            seen = set(new_data[user])  # 只得到item的集合
            pos_num = len(seen)
            items_neg_pool = np.random.choice(items, int(pos_num * ratio * 3), popularitys)
            # items_neg_pool中存在重复值，需要去重后按照ratio比例采样
            items_neg = []
            for item in items_neg_pool:
                if (item not in items_neg) and (item not in seen):
                    items_neg.append(item)
                if len(items_neg) >= int(pos_num * ratio):
                    break
            # item = [x for x in items_neg_pool if x not in seen][:int(pos_num * ratio)]
            new_data[user].update({x: 0 for x in items_neg})
        print('Done with negative sampling.')
        return new_data

    # 训练
    P, Q = {}, {}
    for user in train:
        P[user] = np.random.random(K)
    for item in items:
        Q[item] = np.random.random(K)
    for s in trange(step):  # 梯度下降法更新ini
        data = nSample(train, ratio)
        loss = 0
        for user, items in data.items():
            for item in items:
                eui = data[user][item] - (P[user] * Q[item]).sum()
                loss += eui * eui
                tmp = P[user]
                P[user] += lr * (Q[item] * eui - lmbda * P[user])
                Q[item] += lr * (tmp * eui - lmbda * Q[item])
        for user in train:
            loss += lmbda * np.dot(P[user], P[user])
        for item in items:
            loss += lmbda * np.dot(Q[item], Q[item])
        print('Done training with step={}, loss={}'.format(s + 1, loss))
        if loss < 0.0001:
            break
        lr *= 0.9  # 调整学习率

    # 获取接口函数
    def GetRecommendation(user):
        seen = set(train[user])
        recs = {}
        for item in items:
            if item not in seen:
                recs[item] = (P[user] * Q[item]).sum()
        recs = list( \
            sorted(recs.items(), \
                   key=lambda x: x[1], \
                   reverse=True \
                   ) \
            )[:N]
        return recs

    return GetRecommendation, P, Q
