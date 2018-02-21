import numpy as np
import pandas as pd
import tensorflow as tf


class Data:
    def __init__(self, dataframe):
        self.data = dataframe.copy()
        self.current_ix = 0
        self.length = len(dataframe)
        self.num_epochs = 0

    def shuffle(self):
        raise NotImplementedError()

    def get_next_batch(self, batch_size):
        if (self.length - self.current_ix) < batch_size:
            self.shuffle()
            self.current_ix = 0
            self.num_epochs += 1

        next_batch = self.data.iloc[self.current_ix: self.current_ix + batch_size]
        self.current_ix += batch_size

        return next_batch 


class RepressionData(Data):

    def __init__(self, dataframe):
        super().__init__(dataframe)

    def shuffle(self):
        shuffle_ix = np.random.permutation(self.length)
        self.data = self.data.iloc[shuffle_ix]


class BiochemData(Data):
    def __init__(self, dataframe, cutoff=0.9):
        super().__init__(dataframe)
        self.original_data = dataframe.copy()
        self.original_length = len(self.original_data)
        self.cutoff = cutoff

    def shuffle(self):
        assert (len(self.original_data) == self.original_length)
        self.original_data['keep'] = [(np.random.random() > self.cutoff) if x == 'grey' else True for x in self.original_data['color']]
        self.data = self.original_data[self.original_data['keep']]
        self.length = len(self.data)
        shuffle_ix = np.random.permutation(self.length)
        self.data = self.data.iloc[shuffle_ix]


# class ConvNet:
#     def __init__(self):
#         tf.reset_default_graph()

#         self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#         self._phase_train = tf.placeholder(tf.bool, name='phase_train')

#         self._kd_x = tf.placeholder(tf.float32, shape=[None, 4 * MIRLEN, 4 * SEQLEN, 1], name='kd_x')
#         self._kd_y = tf.placeholder(tf.float32, shape=[None, 1], name='kd_y')
#         self._kd_mask = tf.placeholder(tf.float32, shape=[None, 1], name='kd_mask')
#         self._tpm_mask = tf.placeholder(tf.float32, shape=[None, None, None], name='tpm_mask')
#         self._tpm_y = tf.placeholder(tf.float32, shape=[None, None], name='tpm_y') 

#         # self._freeAGO = tf.get_variable('freeAGO', shape=[1,NUM_TRAIN,1], initializer=tf.constant_initializer(0.0))
#         self._freeAGO = tf.get_variable('freeAGO', shape=[1], initializer=tf.constant_initializer(0.0))
#         self._slope = tf.get_variable('slope', shape=(), initializer=tf.constant_initializer(-0.51023716), trainable=False)

#         self.tensors = {}
#         self.weights = {}
#         self.biases = {}
#         self.latest = None

#     def add_conv_layer(self, layer_name, input, dim1, dim2, stride1, stride2, in_channels, out_channels, phase_train, act):
#         with tf.name_scope(layer_name):
#             _w, _b = helpers.get_conv_params(dim1, dim2, in_channels, out_channels, layer_name)

#             self.weights[layer_name] = _w
#             self.biases[layer_name] = _b

#             _preactivate = tf.nn.conv2d(input, _w1, strides=[1, stride1, stride2, 1], padding='VALID') + _b1
#             _preactivate_bn = tf.layers.batch_normalization(_preactivate, axis=1, training=phase_train)
#             _layer = act(_preactivate_bn)

#             self.tensors[layer_name] = (_preactivate, _preactivate_bn, _layer)
#             self.latest = _layer







