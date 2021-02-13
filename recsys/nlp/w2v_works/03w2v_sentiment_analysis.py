#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/10/29 6:02 PM
    @Author : Caroline
    @File : Word2Vec模型训练词向量，RF模型训练分类器
    @Description : kmeans做feature变换
"""

import os
import re
import logging
import numpy as np
import pandas as pd
import pickle

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

def load_dataset(name, nrows=None):
    datasets = {
        'unlabeled_train': 'unlabeledTrainData.tsv',
        'labeled_train': 'labeledTrainData.tsv',
        'test': 'testData.tsv'
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join('.', 'data', datasets[name])
    df = pd.read_csv(data_file, sep='\t', escapechar='\\', nrows=nrows)
    print('Number of reviews: {}'.format(len(df)))
    return df

def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

# 把原句子中的词的词向量做平均
def to_review_vector(review):
    words = clean_text(review, remove_stopwords = True)
    array = np.array([model[w] for w in words if w in model])
    return pd.Series(array.mean(axis=0))

# 把原句子中的词，统计聚类中心词出现的频数，拉成统一维度，变成features
def make_cluster_bag(review):
    words = clean_text(review, remove_stopwords=True)
    return (pd.Series([word_centroid_map[w] for w in words if w in wordset])
            .value_counts()
            .reindex(range(num_clusters + 1), fill_value=0))

# 在test上做预测，并保存结果
def predict(forest, func = to_review_vector, file_name = 'Word2Vec_model.csv'):
    df = load_dataset('test')
    print('Number of reviews: {}'.format(len(df)))
    test_data_features = df.review.apply(func)
    result = forest.predict(test_data_features)
    output = pd.DataFrame({'id': df.id, 'sentiment': result})
    output.to_csv(os.path.join('.', 'data', file_name), index=False)
    output.head()
    del df
    del test_data_features

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 加载停用词
    eng_stopwords = set(stopwords.words('english'))
    # eng_stopwords = {}.fromkeys([line.rstrip() for line in open('../stopwords.txt')])
    # 加载之前训练好的Word2Vec模型
    model_name = '300features_40minwords_10context.model'
    model = Word2Vec.load(os.path.join('.', 'models', model_name))
    # 加载数据
    df = load_dataset('labeled_train')
    # 将原始数据转换成词向量
    train_data_features = df.review.apply(to_review_vector)
    # 训练分类器
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest = forest.fit(train_data_features, df.sentiment)
    print('Train done.')
    # 清理占用内存的变量
    del df
    del train_data_features
    # 做预测
    predict(forest)
    print('Predict done.')

    # 用用聚类算法
    # 加载已有的词向量
    word_vectors = model.syn1neg
    # 用词向量特征训练kmeans模型
    num_clusters = word_vectors.shape[0] // 10
    print('all words:', len(word_vectors), ', cluster object:', num_clusters)
    kmeans_clustering = KMeans(n_clusters=num_clusters, n_jobs=4)
    idx = kmeans_clustering.fit_predict(word_vectors)
    # 保存模型
    word_centroid_map = dict(zip(model.wv.index2word, idx)) # 获得每个词属于哪个cluster
    filename = 'word_centroid_map_10avg.pickle'
    with open(os.path.join('.', 'models', filename), 'bw') as f:
        pickle.dump(word_centroid_map, f)
    # 加载已有的模型
    # with open(os.path.join('..', 'models', filename), 'br') as f:
    #    word_centroid_map = pickle.load(f)
    # 看看cluster的数据
    for cluster in range(0, 10):
        print("\nCluster %d" % cluster)
        print([w for w, c in word_centroid_map.items() if c == cluster])
    # 清洗数据：对生数据聚类，变换得到统一维度的聚类中心词频数特征
    wordset = set(word_centroid_map.keys())
    df = load_dataset('labeled_train')
    train_data_features = df.review.apply(make_cluster_bag)
    # 喂给RF分类器
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest = forest.fit(train_data_features, df.sentiment)
    print('Train done.')
    # 清理占用内存的变量
    del df
    del train_data_features
    # 做预测
    predict(forest, func = make_cluster_bag, file_name = 'Word2Vec_BagOfClusters.csv')
    print('Predict done.')