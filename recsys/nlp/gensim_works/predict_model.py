#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/10/28 2:38 PM
    @Author : Caroline
    @File : w2v模型的使用
    @Description : 调用w2v模型，查询通过wiki预料训练出的结果
"""
import gensim

if __name__ == '__main__':
    model = gensim.models.Word2Vec.load("wiki.zh.text.model")
    result = model.most_similar(u"足球")
    for e in result:
        print(e[0], e[1])