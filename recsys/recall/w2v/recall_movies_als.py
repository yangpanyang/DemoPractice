#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/10/30 5:04 PM
    @Author : Caroline
    @File : 使用ALS矩阵分解实现U2I召回
    @Description :
        ALS：交替最小二乘法，先假设U的初始值U(0)，可以根据U(0)可以计算出V(0)，再根据V(0)计算出U(1)，迭代到收敛
        矩阵分解：将（用户、物品、行为）矩阵分解成（用户、隐向量）和（物品，隐向量）两个子矩阵，通过隐向量实现推荐
"""

import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.recommendation import ALS
import json
from scipy.spatial import distance  # 余弦相似度


def load_data():
    # 指定excel的解析字段类型
    customSchema = T.StructType([
        T.StructField("userId", T.IntegerType(), True),
        T.StructField("movieId", T.IntegerType(), True),
        T.StructField("rating", T.FloatType(), True),
        T.StructField("timestamp", T.LongType(), True),
    ])
    df = spark.read.csv(ratings_file, header=True, schema=customSchema)
    print('user count: ', df.select("userId").distinct().count())
    print('movie count: ', df.select("movieId").distinct().count())
    # df.printSchema()
    return df


def train_model(df):
    als = ALS(maxIter=5, regParam=0.01,
              userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    # 实现训练
    model = als.fit(df)
    # 保存user隐向量
    model.userFactors.select("id", "features") \
        .toPandas() \
        .to_csv(user_factors_file, index=False)
    print('user factor count: ', model.userFactors.count())
    model.userFactors.show(5)
    # 保存item隐向量
    model.itemFactors.select("id", "features") \
        .toPandas() \
        .to_csv(item_factors_file, index=False)
    print('item factor count: ', model.itemFactors.count())
    model.itemFactors.show(5)


# 实现U2I召回，return: (movieId), {movieId: sim_value}
def get_recommend_results(target_user_id, topK=10):
    df_rating = pd.read_csv(ratings_file)
    # df_movie = pd.read_csv(movies_file)
    df_user_embedding = pd.read_csv(user_factors_file)
    df_movie_embedding = pd.read_csv(item_factors_file)
    # embedding从字符串向量化
    df_user_embedding["features"] = df_user_embedding["features"].map(lambda x: np.array(json.loads(x)))
    df_movie_embedding["features"] = df_movie_embedding["features"].map(lambda x: np.array(json.loads(x)))
    print('user embedding shape: ', df_user_embedding.shape)
    print('item embedding shape: ', df_movie_embedding.shape)
    # 给定target_user_id，使用余弦相似度，计算所有item的similarity score，过滤掉浏览历史，倒排取topK做推荐
    user_embedding = df_user_embedding[df_user_embedding["id"] == target_user_id].iloc[0, 1]
    df_rec = df_movie_embedding.copy()  # 候选集
    df_rec["sim_value"] = df_rec["features"].map(lambda x: 1 - distance.cosine(user_embedding, x))
    # 筛选、查询单列、去重、变成set
    watched_ids = set(df_rating[df_rating["UserID"] == target_user_id]["MovieID"].unique())
    print('view history count: ', len(watched_ids))
    # 过滤浏览历史
    df_rec = df_rec[~df_rec["id"].isin(watched_ids)].sort_values(by="sim_value", ascending=False).reset_index(drop=True)
    df_rec = df_rec[:topK][['id', 'sim_value']]
    rec_movieIds = set(df_rec['id'].values)
    rec_movieId_scores = df_rec.set_index('id').to_dict()['sim_value']
    print('recommend itemIds:', rec_movieIds)
    print('recommend scores', rec_movieId_scores)
    return rec_movieIds, rec_movieId_scores


if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("PySpark ALS") \
        .getOrCreate()
    sc = spark.sparkContext

    ratings_file = os.path.join('../../data/ml-latest-small', 'ratings_1m.csv')
    # movies_file = os.path.join('../../data/ml-latest-small', 'movies.csv')
    user_factors_file = os.path.join('./model', 'movielens_sparkals_user_embedding.csv')
    item_factors_file = os.path.join('./model', 'movielens_sparkals_item_embedding.csv')
    df = load_data()
    train_model(df)
    rec_movieIds, rec_movieId_scores = get_recommend_results(1, 10)
