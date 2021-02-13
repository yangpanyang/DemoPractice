#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/14 10:51 PM
    @Author : Caroline
    @File : 单机estimator实现
    @Description :
"""

import tensorflow as tf
from tensorflow import feature_column

# from tensorflow.keras import layers  # 1.15版本
# from tensorflow.python.feature_column import feature_column_v2 as fc
# tf_single.enable_eager_execution()

columns = [
    'id',
    'click',
    'hour',
    'C1',
    'banner_pos',
    'site_id',
    'site_domain',
    'site_category',
    'app_id',
    'app_domain',
    'app_category',
    'device_id',
    'device_ip',
    'device_model',
    'device_type',
    'device_conn_type',
    'C14',
    'C15',
    'C16',
    'C17',
    'C18',
    'C19',
    'C20',
    'C21',
]


def fc_column(feature_name, hash_bucket_size, dtype=tf.string):
    f = feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size=hash_bucket_size, dtype=dtype)
    f1 = feature_column.embedding_column(f, 10)
    return f1


def fc_transform(feature_name, hash_bucket_size, dtype=tf.string):
    f = feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size=hash_bucket_size, dtype=dtype)
    f1 = feature_column.embedding_column(f, 4)
    feature_layer = tf.keras.layers.DenseFeatures([f1])
    # feature_layer = tf.keras.layers.DenseFeatures([fc_column(feature_name, hash_bucket_size, dtype)])
    return feature_layer


feature_columns = [fc_column('device_ip', 100), fc_column('C1', 100, dtype=tf.int64)]


### Tensorflow have three levels of API:
# Low level API: tf.reduce_sum, tf.matmul
# Mid level API: layers, tf.keras.layers. ps: Dense, Concatenate, Customize a keras layers
    # 极大地解放了生产力，不用写底层算子操作
# High level API: Estimator. ps: Session and Graph
    # session: 拿到当前tennsorflow计算的session去跑op
    # tensorflow需要做图冻结，冻结完之后才开始计算
    # Adavantages: without session, you can focus on model logic
    # Disadvantages: you have loss control with the model, Hooks--到达固定步数的时候做一些操作

# loss、auc等用低阶API，模型具体网络结构用中阶API

# 返回一个函数
def input_fn(file_path):
    dataset = tf.data.experimental.make_csv_dataset(file_path,
                                                    batch_size=8,
                                                    column_names=columns,
                                                    label_name='click',
                                                    na_value="?",
                                                    num_epochs=1)
    dataset = dataset.shuffle(500)  # 打散数据集，参数越大混乱程度也越大
    return dataset.make_one_shot_iterator().get_next()  # 每次返回一个 batch size


# canned estimator
tf.logging.set_verbosity(tf.logging.INFO)  # 打日志: DEBUG, INFO, WARN, ERROR, FATAL
# estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# estimator.train(input_fn=lambda: input_fn("../data/avazu-ctr-prediction/train"), steps=2000)

# customized estimator自定义Estimator：
# model_fn: 模型主体部分，定义了模型的三个部分
    # features: dict of tensors
    # labels: tensor
    # mode: three modes. 模型训练的三个阶段
    # params: 把一些参数放到模型里面去用，为了解耦
# model_dir: 模型训练结果存放的位置，包括event、checkpoint数据
    # Event：给Tensorboard用的，用来画模型的相关指标
    # Checkpoint：tensorflow的数据结构，指定模型全部的数据
        # 正向传播有params
        # 反向传导有gradients，供模型计算的时候使用
    # SavedModel：只保留params，供线上serving的时候使用
    # 每次会检测是否有model_dir，检测是否有checkponts，如果有会load
        # load successful：模型一致才可
        # load failure
# config: 用RunConfig声明分布式的东西、checkpoint的一些逻辑等
# params: 写一些参数透传到model_fn

def model_fn(features, labels, mode, params):
    # API拿到global_step值，单机无所谓，分布式有一些特殊逻辑的时候需要，比如，打一些日志
    global_step = tf.train.get_global_step()
    device_ip = fc_transform('device_ip', 100)(features)
    C1 = fc_transform('C1', 100, dtype=tf.int32)(features)
    # C14 = fc_transform('C1', 100, dtype=tf_single.int32)(features)

    # 定义模型结构
    with tf.variable_scope("ctr"):
        t1 = tf.keras.layers.Concatenate(axis=-1)([device_ip, C1])
        t2 = tf.keras.layers.Dense(4, activation='relu')(t1)
        ctr_logits = tf.keras.layers.Dense(1)(t2)
    ctr_predicted_logit = tf.nn.sigmoid(ctr_logits)

    # Three Modes (Train, Eval, Predict)
    # Predict Spec, 3个mode共用的EstimatorSpec，根据需要返回什么往里填
    # 做predict的时候不传label，也不需要label
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'ctr': ctr_predicted_logit})

    # Eval Spec，定义一些评估指标
    # estimator会自动的把summary merge到一起，在tensorboard展示
    with tf.name_scope('loss'):
        # 计算 logits 与 label 的交叉熵
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=tf.squeeze(ctr_logits)))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('accuracy'):
        ctr_acc = tf.metrics.accuracy(labels=labels, predictions=ctr_predicted_logit, name='ctr_acc')
        tf.summary.scalar('accuracy', ctr_acc[1])
    with tf.name_scope('auc'):
        ctr_auc = tf.metrics.auc(labels=labels, predictions=ctr_predicted_logit, name='ctr_auc')
        tf.summary.scalar('auc', ctr_auc[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops={'accuracy/accuracy': ctr_acc},
                                          evaluation_hooks=None)  # 拿什么做eval_metric_ops，也会打到tensorboard

    # Train Spec: 涉及到模型更新操作
    if params['optimizer'] == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer()
    else:
        optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=global_step)
    # Create a hook to print acc, loss & global step every 100 iter.
    train_hook_list = []
    train_tensors_log = {'ctr_auc': ctr_auc[1], 'loss': loss, 'global_step': global_step}
    train_hook_list.append(
        tf.estimator.CheckpointSaverHook(save_steps=1000, checkpoint_dir="../model_dir/single_estimator"))
    train_hook_list.append(tf.train.LoggingTensorHook(tensors=train_tensors_log, every_n_iter=10))
    # train_op实现真正的训练
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[])  # 中间需要做什么操作的时候塞到training_hooks

### ESSM 模型结构
def model_fn_essm(features, labels, mode, params):
    # API拿到global_step值，单机无所谓，分布式有一些特殊逻辑的时候需要，比如，打一些日志
    global_step = tf.train.get_global_step()
    device_ip = fc_transform('device_ip', 100)(features)
    C1 = fc_transform('C1', 100, dtype=tf.int32)(features)
    # C14 = fc_transform('C1', 100, dtype=tf_single.int32)(features)

    # 定义模型结构
    with tf.variable_scope("ctr"):
        t1 = tf.keras.layers.Concatenate(axis=-1)([device_ip, C1])
        t2 = tf.keras.layers.Dense(4, activation='relu')(t1)
        ctr_logits = tf.keras.layers.Dense(1)(t2)
    with tf.variable_scope("cvr"):
        t3 = tf.keras.layers.Concatenate(axis=-1)([device_ip, C1])
        t4 = tf.keras.layers.Dense(4, activation='relu')(t3)
        cvr_logits = tf.keras.layers.Dense(1)(t4)
    ctr_predicted_logit = tf.nn.sigmoid(ctr_logits)
    cvr_predicted_logit = tf.nn.sigmoid(cvr_logits)

    # Three Modes (Train, Eval, Predict)
    # Predict Spec, 3个mode共用的EstimatorSpec，根据需要返回什么往里填
    # 做predict的时候不传label，也不需要label
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'cvr': cvr_predicted_logit})

    # Eval Spec，定义一些评估指标
    # estimator会自动的把summary merge到一起，在tensorboard展示
    with tf.name_scope('loss'):
        # labels是个tensor，计算 logits 与 label 的交叉熵
        ctr_cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels[:, 0], tf.float32), logits=tf.squeeze(ctr_logits)))
        # 因为预测的 ctcvr = ctr * cvr，但是计算loss是在logits的基础、而不是prediction的基础上算的
        ctcvr_cross_entropy = tf.keras.backend.binary_crossentropy(ctr_predicted_logit*cvr_predicted_logit, tf.cast(labels[:, 1], tf.float32))
        loss = 1.0 * ctr_cross_entropy + 1.0 * ctcvr_cross_entropy  # 权重相加
        tf.summary.scalar('loss', loss)
    with tf.name_scope('accuracy'):
        ctr_acc = tf.metrics.accuracy(labels=labels[:, 0], predictions=ctr_predicted_logit, name='ctr_acc')
        tf.summary.scalar('ctr_acc', ctr_acc[1])
        cvr_acc = tf.metrics.accuracy(labels=labels[:, 1], predictions=cvr_predicted_logit, name='cvr_acc')
        tf.summary.scalar('cvr_acc', cvr_acc[1])
    with tf.name_scope('auc'):
        ctr_auc = tf.metrics.auc(labels=labels, predictions=ctr_predicted_logit, name='ctr_auc')
        tf.summary.scalar('ctr_auc', ctr_auc[1])
        cvr_auc = tf.metrics.auc(labels=labels, predictions=cvr_predicted_logit, name='cvr_auc')
        tf.summary.scalar('cvr_auc', cvr_auc[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops={'accuracy/ctr_acc': ctr_acc,
                                                           'accuracy/cvr_acc': cvr_acc},
                                          evaluation_hooks=None)  # 拿什么做eval_metric_ops，也会达到tensorboard

    # Train Spec: 涉及到模型更新操作
    if params['optimizer'] == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer()
    else:
        optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=global_step)
    # Create a hook to print acc, loss & global step every 100 iter.
    train_hook_list = []
    train_tensors_log = {'ctr_auc': ctr_auc[1], 'cvr_auc': cvr_auc[1], 'loss': loss, 'global_step': global_step}
    train_hook_list.append(
        tf.estimator.CheckpointSaverHook(save_steps=1000, checkpoint_dir="../model_dir/single_estimator"))
    train_hook_list.append(tf.train.LoggingTensorHook(tensors=train_tensors_log, every_n_iter=10))
    # train_op实现真正的训练
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[])  # 中间需要做什么操作的时候塞到training_hooks

# Checkpoints
# Need to talk three points:
# 1, checkpoints structure;
# 2, reload from a checkpoint as latest;
# 3, checkpoints must share the same network structure


# RunConfig：运行的时候的一些参数
    # model_dir
    # save_summary_steps：summary writer往里面写summary，给Estimator使用
    # save_checkpoints_secs：多少秒做一次checkpoint
    # save_checkpoints_steps：多少步做一次checkpoint，与上面只有一个生效
    # keep_checkpoint_max：保留最近N个checkpoint
    # keep_checkpoint_every_n_hours：最近多少步保留checkpoint
    # log_step_count_steps：多少步打一次log
    # train_distribute：分布式相关
    # eval_distribute：分布式相关
checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs=5,  # Save checkpoints every * secs.
    keep_checkpoint_max=5,  # Retain the 10 most recent checkpoints.
)

estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="../model_dir/single_estimator",
                                   config=checkpointing_config,
                                   params={'optimizer': 'estimator'})
# estimator.train(input_fn=lambda: input_fn("../data/avazu-ctr-prediction/train"), max_steps=20000)


### evaluate
# metrics = estimator.evaluate(input_fn=lambda: input_fn("avazu-ctr-prediction/train", 10))


### Now serving
# 作为一个模型，什么时候生产出一个model
# 训练阶段很多指标metrics是不准的，所以要做evaluation，模型是在evaluation阶段生成的
feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
# Another way
# def serving_input_receiver_fn():
#     receiver_tensors = {
#         'device_ip': tf.placeholder(tf.string, [None, 1]),
#         'C1': tf.placeholder(tf.int64, [None, 1]),
#     }
#
#     # Convert give inputs to adjust to the model.
#     # features = {"examples": tf_single.concat([receiver_tensors['device_ip'], receiver_tensors['C1']], axis=1)}
#     return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=receiver_tensors)

# EvalSpec:
    # eval和test数据集不一样
    # throttle_secs 距离上一次evaluate间隔多久，单位s
    # exporters 生成一个模型的exporter
best_exporter = tf.estimator.BestExporter(serving_input_receiver_fn=serving_input_receiver_fn, exports_to_keep=1)
exporters = [best_exporter]  # 调用serving_input_receiver_fn转成exporters存进去
tf.estimator.train_and_evaluate(estimator,
                                train_spec=tf.estimator.TrainSpec(
                                    input_fn=lambda: input_fn("../data/avazu-ctr-prediction/train"),
                                    max_steps=10000),
                                eval_spec=tf.estimator.EvalSpec(
                                    input_fn=lambda: input_fn("../data/avazu-ctr-prediction/train"),
                                    exporters=exporters,
                                    throttle_secs=10)
                                )

### another type to define serving，把模型读进来，放进到固定目录下
export_dir = estimator.export_savedmodel('../model_dir/single_estimator', serving_input_receiver_fn)

# 用下面的命令检视模型，会声明inputs、outputs
# saved_model_cli show --dir ./model_dir/single_estimator/1611506615 --tag_set serve --signature_def serving_default
# saved_model_cli run  --dir ./model_dir/single_estimator/1611506615 --tag_set serve --signature_def serving_default --input_examples 'examples=[{"C1":[12344],"device_ip":[b"1"]}]'



import pandas as pd
# Test inputs represented by Pandas DataFrame.
inputs = pd.DataFrame({
    'device_ip': [b"12312342", b"12312343"],
    'C1': [122, 145],
})
# Convert input data into serialized Example strings.
examples = []
for index, row in inputs.iterrows():
    feature = {}
    for col, value in row.iteritems():
        if col == "device_ip":
            feature[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        if col == "C1":
            feature[col] = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    examples.append(example.SerializeToString())

### 实现线上serving
# 把模型加载到内存
predict_fn = tf.contrib.predictor.from_saved_model(export_dir)
# Make predictions.
predictions = predict_fn({'examples': examples})
print(predictions)

# docker run -t --rm -p 8501:8501 \
#     -v "/Users/sierra/兼职/July/项目/avazu/model_dir/1588011605:/models/half_plus_two" \
#     -e MODEL_NAME=half_plus_two \
#     tensorflow/serving &

# saved_model_cli show --dir ./model_dir/1589123597 --all
# curl http://$(docker-machine ip default):8501/v1/models/half_plus_two/metadata
# curl -d '{"instances":[{"C1": [12344], "device_ip":["1"]}]}' -X POST http://$(docker-machine ip default):8501/v1/models/half_plus_two:predict
# curl -d '{"inputs":{"C1": [[12345]], "device_ip":[["2"]]}}' -X POST http://$(docker-machine ip default):8501/v1/models/half_plus_two:predict
