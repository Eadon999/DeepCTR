from deepctr.models.esmm import ESMM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf

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
    multi_label = [label_ctr, label_cvr]

    ctcvr = ESMM(linear_feature_columns, dnn_feature_columns)
    ctcvr_model = ctcvr.build()
    opt = tf.optimizers.Adam(lr=0.003, decay=0.0001)
    ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
                        metrics=[tf.keras.metrics.AUC()])

    history = ctcvr_model.fit(train_model_input, multi_label, epochs=10, verbose=1, batch_size=10)

    # TODO： TFRecoed to train
