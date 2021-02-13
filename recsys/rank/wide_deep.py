#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/1 1:48 PM
    @Author : Caroline
    @File : keras实现wide&deep模型
    @Description : 用keras搭建网络结果，注意dense层、sparse层变换
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import tensorflow as tf
import tensorflow.keras.backend as K
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import *
from tensorflow.python.keras import *
from tensorflow.python.keras.layers import Flatten, Embedding, concatenate, Dense, Dropout, Activation, \
    BatchNormalization
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


# 用keras搭建 wide&deep 模型
def wd_model(sparse_columns, dense_columns, train, test, lmbda=0.05):
    ####### sparse features ##########
    sparse_input = []
    lr_embedding = []
    for col in sparse_columns:
        ## lr_embedding
        _input = Input(shape=(1,))
        sparse_input.append(_input)

        nums = pd.concat((train[col], test[col])).nunique() + 1  # 取出列col的数据中，去重后的数据类别数
        embed = Flatten()(Embedding(nums, 4, input_length=1, embeddings_regularizer=tf.keras.regularizers.l2(lmbda))(
            _input))  # shape=(None, 1, 4) -> shape=(None, 4)
        lr_embedding.append(embed)

    fst_order_sparse_layer = concatenate(lr_embedding)

    ####### dense features ##########
    dense_input = []
    for col in dense_columns:
        _input = Input(shape=(1,))
        dense_input.append(_input)
    concat_dense_input = concatenate(dense_input)
    fst_order_dense_layer = Dense(4, activation='relu', kernel_regularizer=regularizers.l2(lmbda))(concat_dense_input)

    ####### linear concat ##########
    linear_part = concatenate([fst_order_dense_layer, fst_order_sparse_layer])

    #######dnn layer##########
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(128)(linear_part))))
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(64)(fc_layer))))
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(32)(fc_layer))))

    ######## output layer ##########
    print('linear layer: ', linear_part)
    print('dnn layer: ', fc_layer)
    output_layer = concatenate([linear_part, fc_layer])
    print('hidden layer to output: ', output_layer)
    output_layer = Dense(1, activation='sigmoid')(output_layer)

    model = Model(inputs=sparse_input+dense_input, outputs=output_layer)

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
    # lmbda=0.005, auc=0.77208
    model = wd_model(sparse_cols, dense_cols, x_train, x_valid, lmbda=0.005)
    model.compile(optimizer='adam',
                  loss="binary_crossentropy",
                  metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

    # 训练模型
    file_path = "./models/wide&deep_model.h5"  # 回调函数
    earlystopping = EarlyStopping(monitor="val_loss", patience=3)  # 早停机制
    checkpoint = ModelCheckpoint(
        file_path, save_weights_only=True, verbose=1, save_best_only=True)
    callbacks_list = [earlystopping, checkpoint]

    hist = model.fit(train_sparse_x + train_dense_x,
                     train_label,
                     batch_size=1024, #512,
                     epochs=20,
                     validation_data=(valid_sparse_x + valid_dense_x, valid_label),
                     callbacks=callbacks_list,
                     shuffle=False)

    # 模型评估结果
    print('val-loss: ', np.min(hist.history['val_loss']))
    print('val-auc: ', np.max(hist.history['val_auc']))


def test_model():
    s_cols = ['C1', 'C2', 'C3']
    d_cols = ['I1', 'I2']
    # sparse
    print('----------- sparse features ------------')
    sparse_input = []
    lr_embedding = []
    for s_col in s_cols:
        _input = Input(shape=(1,))
        sparse_input.append(_input)
        print('sparse input: ', sparse_input)
        nums = pd.concat((x_train[s_col], x_valid[s_col])).nunique() + 1
        embed = Flatten()(Embedding(nums, 4, input_length=1, embeddings_regularizer=tf.keras.regularizers.l2(0.5))(
            _input))  # shape=(None, 1, 4) -> shape=(None, 4)
        lr_embedding.append(embed)
    print('\nsparse emb: ', lr_embedding)
    fst_order_sparse_layer = concatenate(lr_embedding)
    print('\nsparse layer: ', fst_order_sparse_layer)
    # dense
    print('----------- dense features ------------')
    dense_input = []
    for d_col in d_cols:
        _input = Input(shape=(1,))
        dense_input.append(_input)
        print('dense input: ', dense_input)
    concat_dense_input = concatenate(dense_input)
    print('\nfinal dense input: ', concat_dense_input)
    fst_order_dense_layer = Dense(4, activation='relu')(concat_dense_input)
    print('\ndense layer: ', fst_order_dense_layer)

    # linear concat
    print('----------- linear part ------------')
    linear_part = concatenate([fst_order_dense_layer, fst_order_sparse_layer])
    print('linear part: ', linear_part)

    #######dnn layer##########
    print('----------- dnn part ------------')
    print('dnn input layer: ', linear_part)
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(128)(linear_part))))
    print('full conection layer1: ', fc_layer)
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(64)(fc_layer))))
    print('full conection layer2: ', fc_layer)
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(32)(fc_layer))))
    print('full conection layer3: ', fc_layer)

    ######## output layer ##########
    print('----------- output part ------------')
    print('linear layer: ', linear_part)
    print('dnn layer: ', fc_layer)
    output_layer = concatenate([linear_part, fc_layer])
    print('hidden layer to output: ', output_layer)
    output_layer = Dense(1, activation='sigmoid')(output_layer)
    print('output layer: ', output_layer)

    ######## model ##########
    print('----------- model ------------')
    model = Model(inputs=sparse_input + dense_input, outputs=output_layer)
    print('input: ', sparse_input + dense_input)
    print('\noutput: ', output_layer)
    print('\nmodel: ', model)
