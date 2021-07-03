import tensorflow as tf

from tensorflow.python.ops.parsing_ops import FixedLenFeature
from deepctr.estimator import DeepFMEstimator
from deepctr.estimator.inputs import input_fn_tfrecord
from deepctr.models import DeepFM

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import pandas as pd

if __name__ == "__main__":

    # 1.generate feature_column for linear part and dnn part

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    dnn_feature_columns = []
    linear_feature_columns = []

    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, 1000), 4))
        linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, 1000))
    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    # 2.generate input data for model

    # shape=1和shape=[1]不影响程序
    feature_description = {k: FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
    feature_description.update(
        {k: tf.io.FixedLenFeature(dtype=tf.float32, shape=1) for k in dense_features})
    feature_description['label'] = FixedLenFeature(dtype=tf.float32, shape=1)


    def input_fn_tfrecord4keras(filenames, feature_description, label=None, batch_size=256, num_epochs=1,
                                num_parallel_calls=8,
                                shuffle_factor=10, prefetch_factor=1,
                                ):
        def _parse_examples(serial_exmp):
            try:
                features = tf.parse_single_example(serial_exmp, features=feature_description)
            except AttributeError:
                features = tf.io.parse_single_example(serial_exmp, features=feature_description)
            if label is not None:
                # labels = features['label']
                # del features['label']
                labels = features.pop(label)
                return features, labels
            return features

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_examples, num_parallel_calls=num_parallel_calls)
        if shuffle_factor > 0:
            dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)

        dataset = dataset.repeat(num_epochs).batch(batch_size)

        # if prefetch_factor > 0:
        #     dataset = dataset.prefetch(buffer_size=batch_size * prefetch_factor)
        return dataset


    train_model_input = input_fn_tfrecord('./criteo_sample.tr.tfrecords', feature_description, 'label', batch_size=256,
                                          num_epochs=100, shuffle_factor=10)
    test_model_input = input_fn_tfrecord('./criteo_sample.te.tfrecords', feature_description, 'label',
                                         batch_size=256, num_epochs=100, shuffle_factor=0)

    # 3.Define Model,train,predict and evaluate
    model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns, task='binary',
                            config=tf.estimator.RunConfig(tf_random_seed=2021))

    model.train(train_model_input)
    eval_result = model.evaluate(test_model_input)

    print(eval_result)

    #######################
    train_model_input = input_fn_tfrecord4keras('./criteo_sample.tr.tfrecords', feature_description, 'label',
                                                batch_size=256,
                                                num_epochs=100, shuffle_factor=10)
    test_model_input = input_fn_tfrecord4keras('./criteo_sample.te.tfrecords', feature_description, 'label',
                                               batch_size=256, num_epochs=100, shuffle_factor=0)

    data = pd.read_csv('./criteo_sample.txt')

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    # TODO:=========================可以通过配置文件设置，不用在train脚本中读取tfrecord之前的文件==============================
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    ###########################

    # 4.test
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')

    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy', tf.keras.metrics.AUC()], )

    print("===============tf version:{}================".format(tf.__version__))
    # ==============tf1中必须写steps_per_epoch, 不然validation data loss和auc等指标显示在第一个epoch为0=====================
    history = model.fit(train_model_input, validation_data=test_model_input, epochs=10, verbose=1, steps_per_epoch=5,
                        validation_steps=3)
    pred_ans = model.predict(test_model_input)

    with tf.Session() as sess:
        x, y = tf.data.make_one_shot_iterator(train_model_input).get_next()
        print(x)
        print(y)
