#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/29 10:56 PM
    @Author : Caroline
    @File : #TODO
    @Description : #TODO
"""

# %config ZMQInteractiveShell.ast_node_interactivity='all'
import os
import sys
from offline import SparkSessionBase
from preprocessing import segmentation

from setting.default import CHANNEL_INFO
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import BucketedRandomProjectionLSH
import faiss
import numpy as np


# 如果当前代码文件运行测试需要加入修改路径，避免出现后导包问题
# BASE_DIR = "/Users/yangpan\ 1/works/recsys"  # "/opt/codes/lesson1"
BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(0, os.path.join(BASE_DIR))
print(BASE_DIR)

PYSPARK_PYTHON = "/usr/local/opt/python@3.7/bin/python3"  # "/usr/bin/python3.6"
# 当存在多个版本时，不指定很可能会导致出错
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_172.jdk/Contents/Home/"


# 创建spark环境
class TrainWord2VecModel(SparkSessionBase):
    SPARK_APP_NAME = "Word2Vec"
    SPARK_URL = "yarn"

    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()


# 对向量vector加权
def compute_vector(row):
    return row.article_id, row.channel_id, row.keyword, row.weight * row.vector


# 求向量list的平均值
def compute_avg_vectors(row):
    x = 0
    for i in row.vectors:
        x += i

    # 求平均值
    return row.article_id, row.channel_id, x / len(row.vectors)


def toArray(row):
    return row.article_id, row.channel_id, [float(i) for i in row.vector]


def toVector(row):
    return row.article_id, Vectors.dense(row.vector)


if __name__ == '__main__':
    w2v = TrainWord2VecModel()  # 初始化spark环境

    w2v.spark.sql("use article")
    article_data = w2v.spark.sql("select * from article_data where article_id=18")
    # segmentation实现分词
    words_df = article_data.rdd.mapPartitions(segmentation).toDF(['article_id', 'channel_id', 'words'])
    # 1.拿到分完词后的文章数据words

    # 训练一个频道的模型
    w2v_model = Word2Vec(vectorSize=100, inputCol='words', outputCol='model', minCount=3)  # 直接调用word2vec训练
    model = w2v_model.fit(words_df)
    # model.write().overwrite().save("hdfs://hadoop-master:9000/headlines/models/word2vec_model/channel_18_python.word2vec")
    # wv = Word2VecModel.load("hdfs://hadoop-master:9000/headlines/models/word2vec_model/channel_18_python.word2vec")
    vectors = model.getVectors()
    # print("word2vec vector size: ", len(vectors.head().vector.toArray()))
    # 2.加载某篇文章的模型，得到每个词的向量

    # 获取频道的文章画像，得到文章画像的关键词(接着之前增量更新的文章article_profile)
    # 获取这些文章20个关键词名称，对应名称找到词向量
    article_profile = w2v.spark.sql("select * from article_profile where channel_id=18")
    article_profile.registerTempTable('profile')
    new_sql = """
        select article_id, channel_id, kw as keyword, weight
        from profile
        LATERAL VIEW explode(keyword) AS kw, weight
        """
    keyword_weight = w2v.spark.sql(new_sql)
    # 合并文章关键词与词向量
    _keywords_vector = keyword_weight.join(vectors, vectors.word == keyword_weight.keyword, 'inner')
    # 计算得到文章每个词的向量
    articleKeywordVectors = _keywords_vector.rdd.map(compute_vector).toDF(
        ["article_id", "channel_id", "keyword", "weightingVector"])
    articleKeywordVectors.registerTempTable('temptable')
    # 计算得到文章的平均词向量，作为文章的向量
    new_sql = """
        select article_id, min(channel_id) channel_id, collect_set(weightingVector) vectors 
        from temptable
        group by article_id
        """
    articleKeywordVectors = w2v.spark.sql(new_sql)
    article_vector = articleKeywordVectors.rdd.map(compute_avg_vectors).toDF(['article_id', 'channel_id', 'vector'])
    article_vector = article_vector.rdd.map(toArray).toDF(['article_id', 'channel_id', 'vector'])
    article_vector.write.insertInto("article_vector")
    # 3.计算得到文章的平均词向量，即文章的向量，并写入hive表

    # 将原始array的数据转换成dense数据
    # article_vector = w2v.spark.sql("select * from article_vector")
    train = article_vector.rdd.map(toVector).toDF(['article_id', 'vector'])
    # 计算相似的文章
    brp = BucketedRandomProjectionLSH(inputCol='vector', outputCol='hashes', numHashTables=4.0, bucketLength=10.0)
    model = brp.fit(train)
    similar = model.approxSimilarityJoin(train, train, 2.0, distCol='EuclideanDistance')
    # similar.selectExpr('min(EuclideanDistance)', 'max(EuclideanDistance)', 'avg(EuclideanDistance)').show()
    # similar.select('EuclideanDistance').rdd.max()
    # 4.召回：采用局部敏感哈希，计算得到文章向量的相似度，做i2i的相似文章召回

    # (1)拿到召回池
    article_vector = w2v.spark.sql("select articlevector from article_vector")
    # collect从DAG图计算出结果；dtype强制需要声明，目前只支持float32；article_vector.count()表示样本数
    vector = np.array(article_vector.collect(), dtype=np.dtype('float32')).reshape([article_vector.count(), 100])
    print("article vector shape: ", vector.shape)
    nlist = 5  # number of clusters，根据clusters聚类
    dimension = 100
    # (2)定义faiss对象——精确匹配版本
    quantiser = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)
    # (3)训练faiss
    print("index.is_trained: ", index.is_trained)  # False
    index.train(vector)  # train on the database vectors
    print("before add index.ntotal: ", index.ntotal)  # 0
    index.add(vector)  # add the vectors and update the index
    print("index.is_trained: ", index.is_trained)  # True
    print("before add index.ntotal: ", index.ntotal)  # 200
    # (4)拿faiss的结果实现i2i的相似召回
    nprobe = 2  # find 2 most similar clusters
    n_query = 10
    k = 3  # return 3 nearest neighbours
    np.random.seed(0)  # 让结果可复现
    # 随机生成n_query个向量，作为待匹配的向量
    query_vectors = np.random.random((n_query, dimension)).astype('float32')
    # 召回：拿faiss的结果做全局搜索，每个向量search出k个结果
    distances, indices = index.search(query_vectors, k)
    # 相似度距离
    print("distances:\n", distances)
    # 召回池里总共有article_vector.count()个向量，indices就是从 0 - article_vector.count()-1 编号的向量
    print("indices:\n", indices)
    # 5.召回：采用faiss，做u2i2i的相似文章召回
