#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/29 11:07 PM
    @Author : Caroline
    @File : 预处理模块
    @Description : 分词
"""

import os
import re
import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs

abspath = "../../words/"


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
