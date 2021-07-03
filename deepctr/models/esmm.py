from itertools import chain

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func, combined_dnn_input


class ESMM:
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME],
                 dnn_hidden_units=(128, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary'):
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.fm_group = fm_group
        self.dnn_hidden_units = dnn_hidden_units
        self.l2_reg_linear = l2_reg_linear
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_dnn = l2_reg_dnn
        self.seed = seed
        self.dnn_dropout = dnn_dropout
        self.dnn_activation = dnn_activation
        self.dnn_use_bn = dnn_use_bn
        self.task = task

    def build(self):
        features = build_input_features(self.linear_feature_columns + self.dnn_feature_columns)

        inputs_list = list(features.values())

        linear_logit = get_linear_logit(features, self.linear_feature_columns, seed=self.seed, prefix='linear',
                                        l2_reg=self.l2_reg_linear)

        group_embedding_dict, dense_value_list = input_from_feature_columns(features, self.dnn_feature_columns,
                                                                            self.l2_reg_embedding,
                                                                            self.seed, support_group=True)

        ctr_sigmoid = self.ctr_model(linear_logit, group_embedding_dict, dense_value_list)
        cvr_sigmoid = self.cvr_model(linear_logit, group_embedding_dict, dense_value_list)

        ctcvr_sigmoid = tf.multiply(ctr_sigmoid, cvr_sigmoid)

        model = tf.keras.models.Model(inputs=inputs_list, outputs=[ctr_sigmoid, ctcvr_sigmoid])

        return model

    def ctr_model(self, linear_logit, group_embedding_dict, dense_value_list):
        fm_logit = add_func([FM()(concat_func(v, axis=1))
                             for k, v in group_embedding_dict.items() if k in self.fm_group])
        print("=================================:", group_embedding_dict.items())
        print("=======concat==========:",
              [concat_func(v, axis=1) for k, v in group_embedding_dict.items() if k in self.fm_group])

        for k, v in group_embedding_dict.items():
            print(k, "<------------------>", v)
            print("==========v len:{}===========".format(len(v)))
        print("==========fm group:{}=============".format(self.fm_group))

        dnn_input = combined_dnn_input(list(chain.from_iterable(
            group_embedding_dict.values())), dense_value_list)
        dnn_output = DNN(self.dnn_hidden_units, self.dnn_activation, self.l2_reg_dnn, self.dnn_dropout, self.dnn_use_bn,
                         seed=self.seed)(dnn_input)
        dnn_logit = tf.keras.layers.Dense(1, use_bias=False,
                                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=self.seed))(
            dnn_output)

        final_logit = add_func([linear_logit, fm_logit, dnn_logit])

        output = PredictionLayer(self.task)(final_logit)
        return output

    def cvr_model(self, linear_logit, group_embedding_dict, dense_value_list):
        fm_logit = add_func([FM()(concat_func(v, axis=1))
                             for k, v in group_embedding_dict.items() if k in self.fm_group])
        print("=================================:", group_embedding_dict.items())
        print("=======concat==========:",
              [concat_func(v, axis=1) for k, v in group_embedding_dict.items() if k in self.fm_group])

        for k, v in group_embedding_dict.items():
            print(k, "<------------------>", v)
            print("==========v len:{}===========".format(len(v)))
        print("==========fm group:{}=============".format(self.fm_group))

        dnn_input = combined_dnn_input(list(chain.from_iterable(
            group_embedding_dict.values())), dense_value_list)
        dnn_output = DNN(self.dnn_hidden_units, self.dnn_activation, self.l2_reg_dnn, self.dnn_dropout, self.dnn_use_bn,
                         seed=self.seed)(dnn_input)
        dnn_logit = tf.keras.layers.Dense(1, use_bias=False,
                                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=self.seed))(
            dnn_output)

        final_logit = add_func([linear_logit, fm_logit, dnn_logit])

        output = PredictionLayer(self.task)(final_logit)
        return output


if __name__ == '__main__':
    linear_feature_columns, dnn_feature_columns = [], []
    ctcvr = ESMM(linear_feature_columns, dnn_feature_columns)
    ctcvr_model = ctcvr.build()
    opt = tf.optimizers.Adam(lr=0.003, decay=0.0001)
    ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
                        metrics=[tf.keras.metrics.AUC()])
