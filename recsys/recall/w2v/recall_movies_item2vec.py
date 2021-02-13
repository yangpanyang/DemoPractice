#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/10/30 3:42 PM
    @Author : Caroline
    @File : 用item2vec实现I2I召回
    @Description : 准备用户浏览数据，作为输入序列，用Word2Vec模型训练item的向量，计算向量相似度，倒排取topK做推荐
"""

import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import Word2Vec
import json
from scipy.spatial import distance  # 余弦相似度


def load_data():
    ratings_file = os.path.join('../../data/ml-latest-small', 'ratings.csv')
    df = pd.read_csv(ratings_file)
    # 只取平均分以上的数据，作为喜欢的列表
    df = df[df["rating"] > df["rating"].mean()].copy()
    # 聚合得到userId，movieId列表
    df_group = df.groupby(['userId'])['movieId'].apply(lambda x: ' '.join([str(m) for m in x])).reset_index()
    uid_movieids_file = os.path.join('./data', 'movielens_uid_movieids.csv')
    df_group.to_csv(uid_movieids_file, index=False)
    print('Trans dataset done.')
    df = spark.read.csv(uid_movieids_file, header=True)
    # 把非常的字符串格式变成list形式
    df = df.withColumn('movie_ids', F.split(df.movieId, " "))
    return df


def train_model(df):
    # 训练模型: https://spark.apache.org/docs/2.4.6/ml-features.html#word2vec
    word2Vec = Word2Vec(vectorSize=5, minCount=0, inputCol="movie_ids", outputCol="movie_2vec")
    # 计算item的embedding
    model = word2Vec.fit(df)
    # 看看训练结果
    model.getVectors().show(3, truncate=False)
    movie_embedding_file = os.path.join('./model', 'movielens_movie_embedding.csv')
    model.getVectors().select("word", "vector") \
        .toPandas() \
        .to_csv(movie_embedding_file, index=False)


# 实现I2I召回，return: (movieId), {movieId: sim_value}
def get_recommend_results(movie_id, topK=10):
    movie_embedding_file = os.path.join('./model', 'movielens_movie_embedding.csv')
    df_embedding = pd.read_csv(movie_embedding_file)
    movies_file = os.path.join('../../data/ml-latest-small', 'movies.csv')
    df_movie = pd.read_csv(movies_file)
    df_merge = pd.merge(left=df_embedding, right=df_movie, left_on="word", right_on="movieId")
    df_merge["vector"] = df_merge["vector"].map(lambda x: np.array(json.loads(x)))
    print('movie embedding shape: ', df_embedding.shape)
    print('movies shape: ', df_movie.shape)
    print('movie embedding merged movies shape: ', df_merge.shape)
    # 给定item，使用余弦相似度，计算所有item的similarity score，倒排取topK做推荐
    movie_embedding = df_merge.loc[df_merge["movieId"] == movie_id, "vector"].iloc[0]
    df_rec = df_merge[df_merge["movieId"] != movie_id]  # 候选集
    df_rec["sim_value"] = df_rec["vector"].map(lambda x: 1 - distance.cosine(movie_embedding, x))
    df_rec = df_rec.sort_values(by="sim_value", ascending=False).reset_index(drop=True)
    df_rec = df_rec[:topK][['movieId', 'sim_value']]  # df_rec[["movieId", "title", "genres", "sim_value"]]
    rec_movieIds = set(df_rec['movieId'].values)
    rec_movieId_scores = df_rec.set_index('movieId').to_dict()['sim_value']
    print('recommend itemIds:', rec_movieIds)
    print('recommend scores', rec_movieId_scores)
    return rec_movieIds, rec_movieId_scores


if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("PySpark Item2vec") \
        .getOrCreate()
    sc = spark.sparkContext

    df = load_data()
    train_model(df)
    rec_movieIds, rec_movieId_scores = get_recommend_results(4018, 10)
