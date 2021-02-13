#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/10/29 5:19 PM
    @Author : Caroline
    @File : Word2Vec模型的使用
    @Description : 主要使用了unlabeled的数据，分词得到词序列，使用Word2Vec训练模型
"""

import os
import re
import logging
import pandas as pd
from bs4 import BeautifulSoup
import nltk.data
from nltk.corpus import stopwords
#nltk.download()
#from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec

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
    return words # ' '.join(words)

# 装饰器模式：统计到一定次数，输出1条标记
def print_call_counts(f):
    n = 0
    def wrapped(*args, **kwargs):
        nonlocal n
        n += 1
        if n % 10000 == 0:
            print('method {} called {} times'.format(f.__name__, n))
        return f(*args, **kwargs)
    return wrapped

@print_call_counts
def split_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [clean_text(s) for s in raw_sentences if s]
    return sentences

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #eng_stopwords = set(stopwords.words('english'))
    eng_stopwords = {}.fromkeys([line.rstrip() for line in open('../stopwords.txt')])
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # 加载数据
    df = load_dataset('unlabeled_train')
    # 清洗数据
    # %time # 统计运行时间
    sentences = sum(df.review.apply(split_sentences), []) # 将多个list(List内部的数据不变)一起组成的1个list
    print('{} reviews -> {} sentences'.format(len(df), len(sentences)))
    # 训练模型
    # 设定词向量训练的参数 https://blog.csdn.net/szlcw1/article/details/52751314
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count, 词频少于min_word_count次数的单词会被丢弃掉, 默认值为5
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    print('Training model...')
    # ******************************************************
    model = Word2Vec(sentences, workers = num_workers, \
                size = num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace = True)
    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)
    model.save(os.path.join('.', 'models', model_name))
    # ******************************************************
    # 模型预测
    print(model.doesnt_match("man woman child kitchen".split()))
    print(model.doesnt_match('france england germany berlin'.split()))
    print(model.most_similar("man"))