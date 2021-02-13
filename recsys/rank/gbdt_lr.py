#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/1 10:30 AM
    @Author : Caroline
    @File : gbdt+lr模型实现
    @Description :
        * 先用gbdt自动选择特征，再使用LR打分
        * 注意：分类器模型参数的选择、特征如何构建
    @Data :
        * 标签: 目标变量，指示是否单击广告（1）（0）
        * I1-I13: 总共13列整数特征（主要是计数特征）
        * C1-C26: 共有26列分类特征。这些功能的值已散列到32位上，以实现匿名化目的
"""

import os
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


def process_sparse_feats(data, cols):
    d = data.copy()
    for f in cols:
        d[f] = d[f].fillna('-1')
        label_encoder = LabelEncoder()
        d[f] = label_encoder.fit_transform(d[f])
    return d


if __name__ == '__main__':
    # 加载数据
    data_file = os.path.join('../data', 'criteo_sampled_data.csv')
    data = pd.read_csv(data_file)
    print('raw data shape: ', data.shape)

    # 清洗数据
    cols = data.columns
    dense_cols = [f for f in cols if f[0] == "I"]
    sparse_cols = [f for f in cols if f[0] == "C"]
    data = process_sparse_feats(data, sparse_cols)
    print('gbdt input data shape: ', data.shape)

    # 切分数据集
    x_train = data[:500000]
    y_train = x_train.pop('label')
    x_valid = data[500000:]
    y_valid = x_valid.pop('label')

    # lightgbm训练模型
    n_estimators = 32  # 50
    num_leaves = 64
    # 开始训练gbdt，50颗树，每课树64个叶节点
    model = lgb.LGBMRegressor(objective='binary',
                              subsample=0.8,
                              min_child_weight=0.5,
                              colsample_bytree=0.8,
                              num_leaves=64,
                              learning_rate=0.1,
                              n_estimators=32,
                              random_state=2020)
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_valid, y_valid)],
              eval_names=['train', 'val'],
              eval_metric='binary_logloss',
              categorical_feature=sparse_cols,
              verbose=10)

    # 提取叶子结点
    # 得到每一条训练数据落在了每棵树的哪个叶子结点上
    # pred_leaf = True 表示返回每棵树的叶节点序号
    gbdt_feats_train = model.predict(x_train, pred_leaf=True)
    # 打印结果的 shape：
    print('gbdt output data shape with train: ', gbdt_feats_train.shape)
    # 打印前3个数据：
    print(gbdt_feats_train[:3])
    # 同样要获取测试集的叶节点索引
    gbdt_feats_valid = model.predict(x_valid, pred_leaf=True)

    # 转换为LR模型的输入
    # 将 50(32) 课树的叶节点序号构造成 DataFrame，方便后续进行 one-hot
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(n_estimators)]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_valid_gbdt_feats = pd.DataFrame(gbdt_feats_valid, columns=gbdt_feats_name)
    train_len = df_train_gbdt_feats.shape[0]
    data = pd.concat([df_train_gbdt_feats, df_valid_gbdt_feats])
    print('gbdt output data shape: ', data.shape)
    # 对每棵树的叶节点序号进行 one-hot
    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)
    # 50颗树，各64个叶子结点
    print('lr input data shape: ', data.shape)

    # 切分数据集
    train = data[: train_len]
    valid = data[train_len:]

    # 开始训练lr
    lr = LogisticRegression(C=5, solver='sag')
    lr.fit(train, y_train)
    # 计算交叉熵损失
    train_logloss = log_loss(y_train, lr.predict_proba(train)[:, 1])
    print('tr-logloss: ', train_logloss)
    valid_logloss = log_loss(y_valid, lr.predict_proba(valid)[:, 1])
    print('val-logloss: ', valid_logloss)
    # AUC评估模型
    auc_score = roc_auc_score(y_valid, lr.predict_proba(valid)[:, 1])
    print('val-auc: ', auc_score)
