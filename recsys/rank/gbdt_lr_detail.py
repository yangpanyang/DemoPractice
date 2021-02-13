#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/5 8:24 PM
    @Author : Caroline
    @File : gbdt与lr模型结合的详细版本
    @Description :
        * 丰富了模型从数据清洗、训练、保存等的操作
        * GBDT、LR 分开训练
        * 训练LR模型还是在 train dataset 上训练、更新
        * 对于 test dataset：先predict出GBDT模型结果，再predict出LR模型结果(最终预测值)
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')
import gc

if __name__ == '__main__':
    # 2.1读入dataset，组织成统一的数据集，并保存下来
    path = os.path.join('.', 'data')
    print('read data...')
    df_train = pd.read_csv(os.path.join(path, 'train.csv'))
    df_test = pd.read_csv(os.path.join(path, 'test.csv'))
    print('read data end.')
    # set(list(df_train.columns.values)).difference(set(list(df_test.columns.values))) # Label
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)
    df_test['Label'] = -1
    data = pd.concat([df_train, df_test])
    data = data.fillna(-1)
    data.to_csv(os.path.join(path, 'data.csv'), index=False)
    print('data saved.')
    print('raw data shape: {}'.format(data.shape))

    # 2.2离散型特征：one-hot encoding
    category_feature = ['C'] * 26
    category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)]
    print('category_feature', category_feature)
    # discrite one-hot encoding
    print('begin one-hot...')
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)  # inplace：是否在原始数据上直接drop
        data = pd.concat([data, onehot_feats], axis=1)
    print('one-hot end.')
    print('new data shape: {}'.format(data.shape))

    # 2.3连续型特征：暂不处理
    continuous_feature = ['I'] * 13
    continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)]
    print('continuous_feature', continuous_feature)
    print('new data shape: {}'.format(data.shape))

    # 2.4切分数据集，将训练集氛围训练集、验证集
    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)
    print('train data shape: {}'.format(train.shape))
    print('test data shape: {}'.format(test.shape))
    print('\nsplit train and testset:')
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)
    print('train data shape: {}, {}'.format(x_train.shape, y_train.shape))
    print('validation data shape: {}, {}'.format(x_val.shape, y_val.shape))

    # 3.1喂数据，fit模型
    print('begin train gbdt:')
    gbm = lgb.LGBMRegressor(objective='binary',
                            subsample=0.8,
                            min_child_weight=0.5,
                            colsample_bytree=0.7,
                            num_leaves=100,
                            max_depth=12,
                            learning_rate=0.05,
                            n_estimators=10,
                            )

    gbm.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_names=['train', 'val'],
            eval_metric='binary_logloss',
            # early_stopping_rounds = 100,
            )
    model = gbm.booster_

    # 3.2在原始数据上做predict，并拼接数据
    print('predict to get leaf...')
    gbdt_feats_train = model.predict(train, pred_leaf=True)  # 分别得到每棵子树的预测值
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    print('df_train_gbdt_feats', df_train_gbdt_feats)
    gbdt_feats_test = model.predict(test, pred_leaf=True)
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)
    print('df_test_gbdt_feats', df_test_gbdt_feats)
    print('predict end.')

    print('create new dataset...')
    train = pd.concat([train, df_train_gbdt_feats], axis=1)
    test = pd.concat([test, df_test_gbdt_feats], axis=1)
    data = pd.concat([train, test])
    train_len = train.shape[0]
    del train
    del test
    gc.collect()
    print('gbdt model raw data shape: {}'.format(data.shape))

    # 3.3 转换数据：将gbdt模型的所有子树值做 onehot encoding
    # leafs one-hot
    print('begin one-hot...')
    for col in gbdt_feats_name:
        print('this is feature:', col)
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)
    print('one-hot ending.')
    print('gbdt model data shape: {}'.format(data.shape))

    # 3.4喂数据，fit第二个LR模型
    train = data[: train_len]
    test = data[train_len:]
    del data
    gc.collect()
    print('lr model train data shape: {}'.format(train.shape))
    print('lr model test data shape: {}'.format(test.shape))
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.3, random_state=2018)
    # lr
    print('beging train lr:')
    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    # 3.5测试集上做predict，得到最终结果，并写入log
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('train-logloss: ', tr_logloss)
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('validation-logloss: ', val_logloss)
    print('begin predict:')
    y_pred = lr.predict_proba(test)[:, 1]
    log_file_name = 'log_gbdt+lr_trlogloss_' + str(round(tr_logloss, 4)) + \
                    '_vallogloss_' + str(round(val_logloss, 4)) + '.csv'
    log_path = os.path.join('./log', log_file_name)
    res = pd.read_csv(os.path.join(path, 'test.csv'))
    log = pd.DataFrame({'Id': res['Id'], 'Label': y_pred})
    log.to_csv(log_path, index=False)
    print('write log in path: {}'.format(log_path))
    print('All end.')