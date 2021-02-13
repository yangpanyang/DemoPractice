#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/10/29 5:35 PM
    @Author : Caroline
    @File : 词袋模型训练词向量，RF模型训练分类器
    @Description : 统计高频词，训练词袋模型，喂给RF训练分类器，在test数据集上预测分类结果
"""

import os
import re
from bs4 import BeautifulSoup
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# return: 清洗后的文本数据
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text() # 去掉HTML标签的数据
    text = re.sub(r'[^a-zA-Z]', ' ', text) # 去掉标点的数据
    words = text.lower().split() # 统一小写，纯词列表数据
    words = [w for w in words if w not in eng_stopwords] # 去掉停用词数据
    return ' '.join(words)

# 在test上做预测，并保存结果
def predict(vectorizer, forest):
    datafile = os.path.join('.', 'data', 'testData.tsv')
    df = pd.read_csv(datafile, sep='\t', escapechar='\\')
    print('Number of reviews: {}'.format(len(df)))
    df['clean_review'] = df.review.apply(clean_text)
    test_data_features = vectorizer.transform(df.clean_review).toarray()
    result = forest.predict(test_data_features)
    output = pd.DataFrame({'id': df.id, 'sentiment': result})
    output.to_csv(os.path.join('.', 'data', 'Bag_of_Words_model.csv'), index = False)
    del df
    del test_data_features

if __name__ == '__main__':
    # 加载停用词
    # stopwords = stopwords.words('english')
    stopwords = {}.fromkeys([line.rstrip() for line in open('../stopwords.txt')])
    eng_stopwords = set(stopwords)
    # 加载数据
    datafile = os.path.join('.', 'data', 'labeledTrainData.tsv')
    df = pd.read_csv(datafile, sep='\t', escapechar='\\')
    print('Number of reviews: {}'.format(len(df)))
    # 清洗数据
    df['clean_review'] = df.review.apply(clean_text)
    print('Clean data done.')
    # 统计词频，top5000作为词表
    vectorizer = CountVectorizer(max_features = 5000)
    train_data_features = vectorizer.fit_transform(df.clean_review).toarray() # 改进：把无标记的数据加进来，一起统计出top5K
    print('Word dictionary done.')
    # 训练分类器
    forest = RandomForestClassifier(n_estimators = 100) # 注意：类别性特征用RF并不太好
    forest = forest.fit(train_data_features, df.sentiment) # 把特征、label喂进来，fit模型
    print('Train done.')
    del df
    del train_data_features
    # 做预测
    predict(vectorizer, forest)
    print('Predict done.')