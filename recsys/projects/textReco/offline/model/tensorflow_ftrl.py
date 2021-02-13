import tensorflow as tf

FEATURE_COLUMN = ['channel_id', 'vector', 'user_weights', 'article_weights']


class LrWithFtrl(object):
    def __init__(self):
        pass

    @staticmethod
    def get_tfrecords_data():
        def parse_example_function(exmaple):
            # 定义解析格式，parse_single_example
            features = {'label': tf.FixedLenFeature([], tf.int64), 'feature': tf.FixedLenFeature([], tf.string)}

            label_feature = tf.parse_single_example(exmaple, features)
            # 修改其中的特征类型和形状
            # 解码 [121]
            # feature = tf.reshape(tf.decode_raw(label_feature['feature'], tf.float32), [1, 121])
            f = tf.decode_raw(label_feature['feature'], tf.float64)
            feature = tf.reshape(tf.cast(f, tf.float32), [1, 121])

            # 计算其中向量、用户权重、文章权重的平均值
            channel_id = tf.cast(tf.slice(feature, [0, 0], [1, 1]), tf.int32)
            vector = tf.reduce_sum(tf.slice(feature, [0, 1], [1, 100]), axis=1)
            user_weights = tf.reduce_sum(tf.slice(feature, [0, 101], [1, 10]), axis=1)
            article_weights = tf.reduce_sum(tf.slice(feature, [0, 111], [1, 10]), axis=1)

            # 4个特征值进行名称构造字典
            data = [channel_id, vector, user_weights, article_weights]
            feature_dict = dict(zip(FEATURE_COLUMN, data))

            label = tf.cast(label_feature['label'], tf.int32)

            return feature_dict, label

        # Tfrecord dataset读取数据
        dataset = tf.data.TFRecordDataset(['../../data/train_ctr.tfrecords'])
        # map 解析
        dataset = dataset.map(parse_example_function)
        dataset = dataset.batch(64)
        dataset = dataset.repeat(10)
        return dataset

    def train_eval(self):
        # 指定列特征
        channel_id = tf.feature_column.categorical_column_with_identity('channel_id', num_buckets=25)

        vector = tf.feature_column.numeric_column('vector')
        user_weights = tf.feature_column.numeric_column('user_weights')
        article_weights = tf.feature_column.numeric_column('article_weights')

        columns = [channel_id, vector, user_weights, article_weights]

        # LinearClassifier
        model = tf.estimator.LinearClassifier(feature_columns=columns,
                                              optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
                                                                               l1_regularization_strength=10,
                                                                               l2_regularization_strength=10))
        
        
        model.train(LrWithFtrl.get_tfrecords_data, steps=100)
        result = model.evaluate(LrWithFtrl.get_tfrecords_data)
        print(result)


if __name__ == '__main__':
    lw = LrWithFtrl()
    # print(lw.get_tfrecords_data())
    lw.train_eval()
