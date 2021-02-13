#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/29 9:25 AM
    @Author : Caroline
    @File : 数据清洗
    @Description :
        * (1)使用jieba进行分词、并做停用词处理
        * (2)tf-idf统计词特征
        * (3)textrank统计词特征
        * (4)将数据写入hive表
"""

import os
import sys

import os
import re
import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs

from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import CountVectorizerModel
from pyspark.ml.feature import IDF
from pyspark.ml.feature import IDFModel

BASE_DIR = "/opt/codes/lesson1"
sys.path.insert(0, os.path.join(BASE_DIR))
print(BASE_DIR)

PYSPARK_PYTHON = "/usr/bin/python3.6"
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON
os.environ["JAVA_HOME"] = '/usr/lib/jvm/java-8-openjdk-amd64'
from offline import SparkSessionBase  # import init文件

abspath = "/opt/words/"


# 加载停用词列表
def get_stopwords_list(stopwords_path):
    """返回stopwords列表"""
    stopwords_list = [i.strip()
                      for i in codecs.open(stopwords_path, encoding='utf-8').readlines()]
    return stopwords_list


# 分词实现
def cut_sentence(sentence, stopwords_list):
    """对切割之后的词语进行过滤，去除停用词，保留名词，英文和自定义词库中的词，长度大于2的词"""
    # print(sentence,"*"*100)
    # eg:[pair('今天', 't'), pair('有', 'd'), pair('雾', 'n'), pair('霾', 'g')]
    seg_list = pseg.lcut(sentence)
    seg_list = [i for i in seg_list if i.flag not in stopwords_list]
    filtered_words_list = []
    for seg in seg_list:
        # print(seg)
        if len(seg.word) <= 1:
            continue
        elif seg.flag == "eng":
            if len(seg.word) <= 2:
                continue
            else:
                filtered_words_list.append(seg.word)
        elif seg.flag.startswith("n"):
            filtered_words_list.append(seg.word)
        elif seg.flag in ["x", "eng"]:  # 是自定一个词语、或者是英文单词，可以加些处理
            filtered_words_list.append(seg.word)
    return filtered_words_list


# 将文章数据进行分词处理，得到分词结果
def segmentation(partition):
    # 结巴加载用户词典
    userDict_path = os.path.join(abspath, "ITKeywords.txt")
    jieba.load_userdict(userDict_path)

    # 加载停用词列表
    stopwords_path = os.path.join(abspath, "stopwords.txt")
    stopwords_list = get_stopwords_list(stopwords_path)

    # 对每一行的sentence字段分词，以生成器的方式返回结果
    for row in partition:
        sentence = re.sub("<.*?>", "", row.sentence)  # 替换掉标签数据
        words = cut_sentence(sentence, stopwords_list)
        yield row.article_id, row.channel_id, words


# 重写分词部分
class TextRank(jieba.analyse.TextRank):
    def __init__(self, stopwords_list, window=20, word_min_len=2):
        super(TextRank, self).__init__()
        self.stopwords_list = stopwords_list
        self.span = window  # 窗口大小
        self.word_min_len = word_min_len  # 单词的最小长度
        # 要保留的词性，根据jieba github ，具体参见https://github.com/baidu/lac
        self.pos_filt = frozenset(
            ('n', 'x', 'eng', 'f', 's', 't', 'nr', 'ns', 'nt', "nw", "nz", "PER", "LOC", "ORG"))

    def pairfilter(self, wp):
        """过滤条件，返回True或者False"""
        if wp.flag == "eng":
            if len(wp.word) <= 2:
                return False

        if wp.flag in self.pos_filt and len(wp.word.strip()) >= self.word_min_len \
                and wp.word.lower() not in self.stopwords_list:
            return True


# texrank
def textrank(partition):
    # 结巴加载用户词典
    userDict_path = os.path.join(abspath, "ITKeywords.txt")
    jieba.load_userdict(userDict_path)

    # 加载停用词列表
    stopwords_path = os.path.join(abspath, "stopwords.txt")
    stopwords_list = get_stopwords_list(stopwords_path)

    # TextRank过滤窗口大小为5，单词最小为2
    textrank_model = TextRank(stopwords_list, window=5, word_min_len=2)
    allowPOS = ('n', "x", 'eng', 'nr', 'ns', 'nt', "nw", "nz", "c")

    for row in partition:
        tags = textrank_model.textrank(row.sentence, topK=20, withWeight=True, allowPOS=allowPOS, withFlag=False)
        for tag in tags:
            yield row.article_id, row.channel_id, tag[0], tag[1]


# 创建spark环境
class OriginArticleData(SparkSessionBase):
    SPARK_APP_NAME = "TextProcessing"
    # SPARK_URL = "yarn"

    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()


# 逐行(Row)解析，使用yield，返回所有文章、所有词语的"文章ID-频道ID-词语索引-词语权重"的映射关系
def sort_by_tfidf(partition):
    TOPK = 20
    # 逐行解析，找到每个词语的索引与IDF值，并按照权重倒排
    for row in partition:
        # 一行(Row)有很多个词，所以要用zip展开，分别拿到"索引-权重"的tuple值
        _dict = list(zip(row.idfFeatures.indices, row.idfFeatures.values))
        _dict = sorted(_dict, key=lambda x: x[1], reverse=True)
        result = _dict[:TOPK]  # 一行（一个Row代表了一篇article）返回TOPK个重要的词
        for word_index, tfidf in result:  # 重新拼接值返回，对于当前行(Row)的article_id、channel_id是一样的
            yield row.article_id, row.channel_id, int(word_index), round(float(tfidf), 4)


# 重新组装数据格式[keyword, idf-weight, index]，比如，['this', 0.6061358035703155, 0]
def append_index(data):
    for index in range(len(data)):
        data[index] = list(data[index])  # 将元组转为list，比如('this', 0.6061358035703155)
        data[index].append(index)  # 加入单词的索引index
        data[index][1] = float(data[index][1])  # 转换单词的idf权重值的格式


# 合并关键词和权重到字典
def merge_kwd_weight(row):
    return row.article_id, row.channel_id, dict(zip(row.keywords, row.weights))


if __name__ == '__main__':
    oa = OriginArticleData()  # 初始化spark环境

    # 读取文章数据，进行每篇章分词
    oa.spark.sql("use article")
    article_data = oa.spark.sql("select * from article_data")
    # article_data.show()
    # segmentation实现分词
    words_df = article_data.rdd.mapPartitions(segmentation).toDF(['article_id', 'channel_id', 'words'])
    print("segmentation done")
    # 1.拿到分完词后的文章数据words

    # 先计算分词之后的每篇文章的词频，得到CV模型
    cv = CountVectorizer(inputCol='words', outputCol='countFeatures', vocabSize=1000, minDF=5)
    cv_model = cv.fit(words_df)  # 统计所有文章不同的词，组成一个词列表 words_list = [1,2,3,,34,4,45,56,67,78,8.......,,,,.]
    print("fit count vectorizer done")
    cv_model.write().overwrite().save("hdfs://hadoop-master:9000/headlines/models/test.model")
    cv_model = CountVectorizerModel.load("hdfs://hadoop-master:9000/headlines/models/test.model")
    # cv_model.vocabulary[:10]  # 看看词频模型的top10，['this', 'pa', 'node', 'data', '数据', 'let', 'keys', 'obj', '组件', 'npm']
    cv_result = cv_model.transform(words_df)
    # print("count vectorizer result: ", cv_result.head())
    print("count vectorizer done")
    # 2.拿到统计完词频的值countFeatures：(共有多少个单词(1266), {每个单词的索引(indices): 每个单词的词频(count)})
    # 比如，countFeatures=SparseVector(1266, {0: 5.0})

    # 然后根据词频计算IDF以及词，得到IDF模型
    idf = IDF(inputCol="countFeatures", outputCol="idfFeatures")
    idf_model = idf.fit(cv_result)
    idf_model.write().overwrite().save("hdfs://hadoop-master:9000/headlines/models/testIDF.model")
    idf_model = IDFModel.load("hdfs://hadoop-master:9000/headlines/models/testIDF.model")
    # print("idf value: ", idf_model.idf.toArray()[:10])
    tfidf_res = idf_model.transform(cv_result)  # IDF对CV结果进行计算TF-IDF
    # print("tfidf result: ", tfidf_res.head())
    print("tfidf done")
    # 3.拿到计算完tf-idf的值idfFeatures，比如，idfFeatures=SparseVector(1266, {0: 3.0307})

    # 解析出词表中所有词的idf值[(keyword, idf-weight)]，并写入hive表
    keywords_list_with_idf = list(zip(cv_model.vocabulary, idf_model.idf.toArray()))
    # 重新组装数据格式，变tuple为list[[keyword, idf-weight, index]]，比如，['this', 0.6061358035703155, 0]
    append_index(keywords_list_with_idf)
    rdd = oa.spark.sparkContext.parallelize(keywords_list_with_idf)  # 创建rdd
    idf_keywords = rdd.toDF(["keywords", "idf", "index"])
    # 把生成的idf数据写入hive表中
    # idf_keywords.write.insertInto('idf_keywords_values')  # 数据直接插入hive表，不会删除已有数据
    idf_keywords.createOrReplaceTempView("tmp_idf_keywords")  # 以临时表的形式重新插入数据
    new_sql = """
        insert overwrite table idf_keywords_values
        select *
        from tmp_idf_keywords
        """
    oa.spark.sql(new_sql)
    print("idf_keywords_values done")
    # 4.根据需要的数据格式进行转换，将词表中所有词的tf-idf值写入hive表

    # 逐行(Row)解析，返回所有文章、所有词语的"文章ID-频道ID-词语索引-词语权重"的映射关系
    keywords_by_tfidf = tfidf_res.rdd.mapPartitions(sort_by_tfidf).toDF(
        ['article_id', 'channel_id', 'index', 'weights'])
    """  # 单独测试看看结果
    row = tfidf_res.head(1)[0]
    row.idfFeatures
    _ = list(zip(row.idfFeatures.indices, row.idfFeatures.values))
    _ = sorted(_, key=lambda x: x[1], reverse=True)
    _
    result = _[:20]
    result
    """
    # 5.拿到每篇article、每个word-index的tf-idf值，最多article_num*word_num个结果，比如，10*20=200

    # 找到文章对应的关键词keyword的tf-idf值，并写入hive表
    keywords_index = oa.spark.sql("select keyword, index idx from idf_keywords_values")
    keywords_result = keywords_by_tfidf.join(keywords_index, keywords_by_tfidf.index == keywords_index.idx) \
        .select(["article_id", "channel_id", "keyword", "weights"])
    # keywords_result.write.insertInto("tfidf_keywords_values")
    keywords_result.createOrReplaceTempView("tmp_keywords_result")  # 以临时表的形式重新插入数据
    new_sql = """
        insert overwrite table tfidf_keywords_values
        select *
        from tmp_keywords_result
        """
    oa.spark.sql(new_sql)
    print("tfidf_keywords_values done")
    # 6.根据index拼接上keyword，将tf-idf值写入hive表

    # 将文章数据过textRank，得到新的keyword权重值，并写入hive表
    textrank_keywords_df = article_data.rdd.mapPartitions(textrank) \
        .toDF(["article_id", "channel_id", "keyword", "textrank"])
    # textrank_keywords_df.write.insertInto("textrank_keywords_values")
    textrank_keywords_df.createOrReplaceTempView("tmp_textrank_keywords_df")  # 以临时表的形式重新插入数据
    new_sql = """
        insert overwrite table textrank_keywords_values
        select *
        from tmp_textrank_keywords_df
        """
    oa.spark.sql(new_sql)
    print("textrank_keywords_values done")
    # 7.拿到keyword计算完textRank的值，并写入hive表

    # 计算tfidf与texrank共同词作为主题词topics
    topic_sql = """
        create table if not exists tmp_table_article_topics as
        select t.article_id as article_id2, collect_set(t.keyword) as topics
        from
            tfidf_keywords_values t
            inner join
            textrank_keywords_values r
        where t.keyword=r.keyword
        group by t.article_id
        """
    article_topics = oa.spark.sql(topic_sql)
    print("tmp_table_article_topics done")
    # 8.拿到每篇文章article的主题词topics

    # 文章画像：每篇文章内的关键词权重合并(textrank * idf)
    idf_keywords_values = oa.spark.sql("select * from idf_keywords_values")
    textrank_keywords_df = oa.spark.sql("select * from textrank_keywords_values")
    keywords_res = textrank_keywords_df.join(idf_keywords_values, on=['keyword'], how='left')
    # 更新文章的关键词keyword权重（以textrank计算拿到的关键词为准）
    keywords_weights = keywords_res.withColumn('weights', keywords_res.textrank * keywords_res.idf) \
        .select(["article_id", "channel_id", "keyword", "weights"])
    # 根据文章ID聚合，拿到每篇文章的权重信息
    keywords_weights.registerTempTable('temp')
    new_sql = """
        select article_id, min(channel_id) channel_id, collect_list(keyword) keywords, collect_list(weights) weights
        from temp
        group by article_id
        """
    keywords_weights = oa.spark.sql(new_sql)
    article_kewords = keywords_weights.rdd.map(merge_kwd_weight).toDF(['article_id', 'channel_id', 'keywords'])
    print("merge_kwd_weight done")
    article_kewords.createOrReplaceTempView("tmp_article_kewords")  # 以临时表的形式重新插入数据
    new_sql = """
        create table if not exists tmp_table_article_kewords as
        select *
        from tmp_article_kewords
        """
    oa.spark.sql(new_sql)
    print("tmp_table_article_kewords done")
    # 9.拿到每篇文章article的关键词keywords对应的权重信息

    # 关键词keywords与主题词topics结果合并，得到文章的最终完整画像，并写入hive表
    article_topics = oa.spark.sql("select * from tmp_table_article_topics")
    article_kewords = oa.spark.sql("select * from tmp_table_article_kewords")
    article_profile = article_kewords.join(article_topics, article_kewords.article_id == article_topics.article_id2) \
        .select(["article_id", "channel_id", "keywords", "topics"])
    # article_profile.write.insertInto("article_profile")
    article_profile.createOrReplaceTempView("tmp_article_profile")  # 以临时表的形式重新插入数据
    new_sql = """
        insert overwrite table article_profile
        select *
        from tmp_article_profile
        """
    oa.spark.sql(new_sql)
    print("article_profile done")
    # 10.拿到文章画像profile数据，并写入hive表
