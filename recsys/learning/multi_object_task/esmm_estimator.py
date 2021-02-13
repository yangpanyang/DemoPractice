#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @Time : 2020/11/20 4:43 PM
    @Author : Caroline
    @File : ESMM模型estimator部分实现
    @Description : CTCVR模型
"""

import tensorflow as tf
# from pypai.commons.utils.tensorflow import embedding_utils


def base_model(base_embeddings, mode, params):
    """
    base dnn model
    :param base_embeddings: input embedding tensor
    :param mode: tf.estimator.ModeKeys.EVAL/TRAIN/PREDICT
    :param params: conf get params
    :return: output logits
    """
    net = base_embeddings
    for units in params['base_hidden_units']:
        # net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        net = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)(net)  # 先构层、再传参数
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(net, 1, activation=None)
    return logits


# tf.layers.dense
# tf.layers.Dense
# tf.keras.layers.Dense # tf官方推荐、标准化、可以自定义层


class MyEstimator(tf.estimator.Estimator):
    def __init__(self, model_fn, model_dir=None, config=None, params=None):
        def _model_fn(features, labels, mode, params):
            # estimator part
            with tf.variable_scope('ctr_model'):
                ctr_logits = base_model(features, mode, params)
            with tf.variable_scope('cvr_model'):
                cvr_logits = base_model(features, mode, params)

            ctr_predictions = tf.sigmoid(ctr_logits, name='CTR')
            cvr_predictions = tf.sigmoid(cvr_logits, name='CVR')

            ctcvr_predictions = tf.multiply(ctr_predictions, cvr_predictions, name='CTCVR')

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'ctcvr_predictions': ctcvr_predictions,
                    'ctr_probabilities': ctr_predictions,
                    'cvr_probabilities': cvr_predictions,
                    'kpi_score': ctcvr_predictions
                }
                export_outputs = {'prediction': tf.estimator.export.PredictOutput(predictions)}
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            ctr_y = tf.cast(labels['label'], tf.float32)
            ctcvr_y = tf.cast(labels['second_label'], tf.float32)

            # 能用logit,一定用logit
            ctr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_y, logits=ctr_logits),
                                      name='ctr_loss')  # 对所有样本取平均值
            ctcvr_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(ctcvr_y, ctcvr_predictions),
                                        name='ctcvr_loss')
            loss = tf.add(ctr_loss, ctcvr_loss, name='ctr_loss')

            ctr_accuracy = tf.metrics.accuracy(labels=ctr_y,
                                               predictions=tf.to_float(tf.greater_equal(ctr_predictions, 0.5)))
            ctcvr_accuracy = tf.metrics.accuracy(labels=ctcvr_y,
                                                 predictions=tf.to_float(tf.greater_equal(ctcvr_predictions, 0.5)))

            ctr_auc = tf.metrics.auc(ctr_y, ctr_predictions)
            ctcvr_auc = tf.metrics.auc(ctcvr_y, ctcvr_predictions)

            tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
            tf.summary.scalar('ctr_auc', ctr_auc[1])
            tf.summary.scalar('ctr_loss', ctr_loss)
            tf.summary.scalar('ctcvr_accuracy', ctcvr_accuracy[1])
            tf.summary.scalar('ctcvr_auc', ctcvr_auc[1])
            tf.summary.scalar('ctcvr_loss', ctcvr_loss)
            # summary_op传给hook，就可以使用tensorboard
            summary_op = tf.summary.merge_all()
            summary_hook = tf.train.SummarySaverHook(save_steps=10,
                                                     output_dir="../model_dir/multi_object_task",
                                                     summary_op=summary_op)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])

            metrics = {
                'ctr_accuracy': ctr_accuracy,
                'ctr_auc': ctr_auc,
                'ctcvr_accuracy': ctcvr_accuracy,
                'ctcvr_auc': ctcvr_auc
            }
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        super(MyEstimator, self).__init__(_model_fn, model_dir, config, params)
