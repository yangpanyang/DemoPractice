#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/5 9:57 AM
    @Author : Caroline
    @File : DeepFM模型实现
    @Description : 使用FM+MLP，注意embedding的取值
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import *
from tensorflow.python.keras.layers import Flatten, Embedding, concatenate, Dense, Dropout, Activation, \
    BatchNormalization, Reshape, Lambda, Add, subtract
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import *


# dense特征：做标准化
def process_dense_feats(data, cols):
    d = data.copy()
    for f in cols:
        d[f] = d[f].fillna(0)
        ss = StandardScaler()
        d[f] = ss.fit_transform(d[[f]])
    return d


# sparse特征：转为labelencoder编码
def process_sparse_feats(data, cols):
    d = data.copy()
    for f in cols:
        d[f] = d[f].fillna('-1')
        label_encoder = LabelEncoder()
        d[f] = label_encoder.fit_transform(d[f])
    return d


def deepfm_model(sparse_columns, dense_columns, train, test, lmbda=0.05):
    ####### sparse features ##########
    sparse_input = []
    lr_embedding = []
    fm_embedding = []
    print('----------- sparse features ------------')
    for col in sparse_columns:
        ## lr_embedding
        _input = Input(shape=(1,))
        sparse_input.append(_input)

        nums = pd.concat((train[col], test[col])).nunique() + 1
        embed = Flatten()(Embedding(nums, 1, embeddings_regularizer=tf.keras.regularizers.l2(lmbda))(_input))
        lr_embedding.append(embed)

        ## fm_embedding
        embed = Embedding(nums, 10, input_length=1, embeddings_regularizer=tf.keras.regularizers.l2(lmbda))(_input)
        reshape = Reshape((10,))(embed)
        fm_embedding.append(reshape)
    fst_order_sparse_layer = concatenate(lr_embedding)
    print('sparse input: ', len(sparse_input), sparse_input)
    print('sparse emb: ', len(lr_embedding), lr_embedding)
    print('sparse fm emb: ', len(fm_embedding), fm_embedding)

    ####### fm layer ##########
    print('----------- fm layer ------------')
    fm_square = Lambda(lambda x: K.square(x))(Add()(fm_embedding))  # 先求和，再求平方
    square_fm = Add()([Lambda(lambda x: K.square(x))(embed)
                       for embed in fm_embedding])  # 先求平方，再求和
    snd_order_sparse_layer = Lambda(lambda x: x * 0.5)(subtract([fm_square, square_fm]))
    print('fm square before sum: ', fm_square)
    print('fm sum after square: ', square_fm)
    print('fm subtract to end: ', snd_order_sparse_layer)

    ####### dense features ##########
    print('----------- dense features ------------')
    dense_input = []
    for col in dense_columns:
        _input = Input(shape=(1,))
        dense_input.append(_input)
    concat_dense_input = concatenate(dense_input)
    fst_order_dense_layer = Dense(4, activation='relu')(concat_dense_input)
    print('dense input: ', len(dense_input), dense_input)
    print('final dense input: ', concat_dense_input)
    print('dense layer: ', fst_order_dense_layer)

    ####### linear concat ##########
    print('----------- linear layer ------------')
    linear_part = concatenate([fst_order_dense_layer, fst_order_sparse_layer])
    print('linear part: ', linear_part)

    #######dnn layer##########
    print('----------- dnn layer ------------')
    print('input sparse embedding: ', len(fm_embedding), fm_embedding)
    concat_fm_embedding = concatenate(fm_embedding, axis=-1)  # (None, 10*26)
    concat_fm_embedding = concatenate([fst_order_dense_layer, concat_fm_embedding])  # (None, 10*26+4)，添加dense的emb特征
    print('dnn input layer: ', concat_fm_embedding)
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(128)(concat_fm_embedding))))
    print('full conection layer1: ', fc_layer)
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(64)(fc_layer))))
    print('full conection layer2: ', fc_layer)
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(32)(fc_layer))))
    print('full conection layer3: ', fc_layer)

    ######## output layer ##########
    print('----------- output layer ------------')
    print('linear layer: ', linear_part)
    print('fm layer: ', snd_order_sparse_layer)
    print('dnn layer: ', fc_layer)
    output_layer = concatenate([linear_part, snd_order_sparse_layer, fc_layer])  # (None, )
    print('hidden layer to output: ', output_layer)
    output_layer = Dense(1, activation='sigmoid')(output_layer)
    print('output layer: ', output_layer)

    print('----------- model ------------')
    model = Model(inputs=sparse_input + dense_input, outputs=output_layer)
    print('model: ', model)

    return model


if __name__ == '__main__':
    # 加载数据
    data_file = os.path.join('../data', 'criteo_sampled_data.csv')
    data = pd.read_csv(data_file)
    print('raw data shape: ', data.shape)

    # 清洗数据
    cols = data.columns
    dense_cols = [f for f in cols if f[0] == "I"]
    sparse_cols = [f for f in cols if f[0] == "C"]
    data = process_dense_feats(data, dense_cols)
    data = process_sparse_feats(data, sparse_cols)
    print('clean data shape: ', data.shape)

    # 切分数据集
    x_train = data[:500000]
    y_train = x_train.pop('label')
    x_valid = data[500000:]
    y_valid = x_valid.pop('label')

    # 转换数据集的格式
    train_sparse_x = [x_train[f].values for f in sparse_cols]
    train_dense_x = [x_train[f].values for f in dense_cols]
    train_label = [y_train.values]
    valid_sparse_x = [x_valid[f].values for f in sparse_cols]
    valid_dense_x = [x_valid[f].values for f in dense_cols]
    valid_label = [y_valid.values]

    # 构建模型
    # lmbda=0.01, auc=0.7689
    # lmbda=0.005, auc=0.7729
    model = deepfm_model(sparse_cols, dense_cols, x_train, x_valid, lmbda=0.005)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

    # 训练模型
    file_path = "./models/deepfm_model.h5"  # 回调函数
    earlystopping = EarlyStopping(monitor="val_loss", patience=3)
    checkpoint = ModelCheckpoint(
        file_path, save_weights_only=True, verbose=1, save_best_only=True)
    callbacks_list = [earlystopping, checkpoint]

    hist = model.fit(train_sparse_x + train_dense_x,
                     train_label,
                     batch_size=1024,
                     epochs=20,
                     validation_data=(valid_sparse_x + valid_dense_x, valid_label),
                     callbacks=callbacks_list,
                     shuffle=False)

    # 模型评估结果
    print('val-loss: ', np.min(hist.history['val_loss']))
    print('val-auc: ', np.max(hist.history['val_auc']))
