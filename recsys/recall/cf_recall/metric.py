#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/5 7:59 PM
    @Author : Caroline
    @File : 评估指标类
    @Description : 计算各种实验评估指标的方法实现
"""

import math


# 评估实验结果
class Metric():

    def __init__(self, train, test, GetRecommendation):
        '''
        :params: trian, 训练数据
        :params: test, 测试数据
        :params: GetRecommendation, 为某个用户获取推荐物品的接口函数
        '''
        self.train = train
        self.test = test
        self.GetRecommendation = GetRecommendation
        self.recs = self.getRec()

    # 为test中的每个用户进行推荐
    def getRec(self):
        recs = {}
        for user in self.test:
            recs[user] = self.GetRecommendation(user)
        print('Done with getRecommendation function.')
        return recs

    # 定义精确率指标计算公式
    def precision(self):
        alls, hit = 0, 0
        for user, items in self.test.items():
            test_items = set(items)
            rank = self.recs[user]  # user的推荐结果list
            rec_items = set([rec[0] for rec in rank])  # user的推荐itemId的集合
            hit = len(list(test_items.intersection(rec_items)))
            # for item, score in rank:
            #     if item in test_items:
            #         hit += 1
            alls += len(rank)
        print('Done with precision function.')
        return round(hit / alls * 100, 2)

        # 定义召回率指标计算公式

    def recall(self):
        alls, hit = 0, 0
        for user, items in self.test.items():
            test_items = set(items)
            rank = self.recs[user]  # user的推荐结果list
            rec_items = set([rec[0] for rec in rank])  # user的推荐itemId的集合
            hit = len(list(test_items.intersection(rec_items)))
            # for item, score in self.recs[user]:
            #     if item in test_items:
            #        hit += 1
            alls += len(test_items)
        print('Done with recall function.')
        return round(hit / alls * 100, 2)

    # 定义覆盖率指标计算方式
    def coverage(self):
        all_item, reco_item = set(), set()
        for user in self.test:
            for item in self.train[user]:
                all_item.add(item)
            for item, score in self.recs[user]:
                reco_item.add(item)
        print('Done with coverage function.')
        return round(len(reco_item) / len(all_item) * 100, 2)

    # 定义新颖度指标计算方式
    def popularity(self):
        # 计算物品的流行度
        item_pop = {}
        for user in self.train:
            for item in self.train[user]:
                if item not in item_pop:
                    item_pop[item] = 0
                item_pop[item] += 1
        num, pop = 0, 0
        for user in self.test:
            for item, score in self.recs[user]:
                # 取对数，防止因长尾问题带来的被流行物品所主导
                if item not in item_pop:
                    print('Error with no item in item_pop.')
                pop += math.log(1 + item_pop[item])
                num += 1
        print('Done with popularity function.')
        return round(pop / num * 100, 6)

    # 展开数据
    def eval(self):
        metrics = { \
            'Precision': self.precision(), \
            'Recall': self.recall(), \
            'Coverage': self.coverage(), \
            'Popularity': self.popularity() \
            }
        print('Metric: {}'.format(metrics))
        return metrics
