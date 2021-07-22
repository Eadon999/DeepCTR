from deepctr.models.esmm import ESMM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from tensorflow.python.ops.parsing_ops import FixedLenFeature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


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

            # 多个label在tf1中要放到元组中(list )：return features, (labels, labels)；tf2中不用放元组中直接：return features, labels, labels时 程序正常，但是结果有不准确，用法不对。故：建议用元组方式
            # 如果使用dict则dict的key name要和Model(outputs=[]),中所用的layer name一致
            # return features, {"ctr_output": label_ctr, "ctcvr_score": label_cvr}
            return features, (labels, labels)
        return features

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_examples, num_parallel_calls=num_parallel_calls)
    if shuffle_factor > 0:
        dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)

    dataset = dataset.repeat(num_epochs).batch(batch_size)

    # if prefetch_factor > 0:
    #     dataset = dataset.prefetch(buffer_size=batch_size * prefetch_factor)
    return dataset


if __name__ == '__main__':
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

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

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    label_ctr = train[target].values
    # 随机生成cvr label
    label_cvr = np.random.randint(0, 2, len(label_ctr))
    multi_label = [label_ctr, label_ctr]

    ctcvr = ESMM(linear_feature_columns, dnn_feature_columns)
    ctcvr_model = ctcvr.build()

    try:
        opt = tf.train.AdamOptimizer()

    except:
        opt = tf.optimizers.Adam()
    ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
                        metrics=[tf.keras.metrics.AUC()])

    history = ctcvr_model.fit(train_model_input, multi_label, epochs=10, verbose=1, batch_size=10)

    # TODO： TFRecoed to train
    # =======================================tf record============================================
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

    # shape=1和shape=[1]不影响程序
    feature_description = {k: FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
    feature_description.update(
        {k: tf.io.FixedLenFeature(dtype=tf.float32, shape=1) for k in dense_features})
    feature_description['label'] = FixedLenFeature(dtype=tf.float32, shape=1)

    #######################
    train_model_input = input_fn_tfrecord4keras('./criteo_sample.tr.tfrecords', feature_description,
                                                'label',
                                                batch_size=256,
                                                num_epochs=100, shuffle_factor=10)

    test_model_input = input_fn_tfrecord4keras('./criteo_sample.te.tfrecords', feature_description,
                                               'label',
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
    print("=================={}\n\n".format(train_model_input))
    # 4.test
    ctcvr = ESMM(linear_feature_columns, dnn_feature_columns)
    ctcvr_model = ctcvr.build()
    try:
        opt = tf.train.AdamOptimizer()
    except:
        opt = tf.optimizers.Adam()
    ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
                        metrics=[tf.keras.metrics.AUC()])

    checkpoint_callback = ModelCheckpoint('./test.h5', verbose=1)
    tensorboard_logs = TensorBoard(log_dir='./logs')
    # ==============tf1中必须写steps_per_epoch, 不然validation data loss和auc等指标显示在第一个epoch为0=====================
    history = ctcvr_model.fit(train_model_input, epochs=10, verbose=1,
                              callbacks=[checkpoint_callback, tensorboard_logs])

    ctcvr_model.save('./model')

    pred_ans = ctcvr_model.predict(train_model_input)
    print(pred_ans)
    print(ctcvr_model.output_names)
    print(ctcvr_model.summary())
    # dot_img_file = '/tmp/model_1.png'
    # tf.keras.utils.plot_model(ctcvr_model, to_file=dot_img_file, show_shapes=True)

    print("===================test take========================")
    # for i in train_model_input.take(1):
    #     print(i)
    print("=====================================================")
