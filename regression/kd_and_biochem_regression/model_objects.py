import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import config, helpers


class Model:
    def __init__(self, mirlen, seqlen, num_train, logdir, baselines, sess):

        self.MIRLEN = mirlen
        self.SEQLEN = seqlen
        self.NUM_TRAIN = num_train
        self.LOGDIR = logdir
        self.SESS = sess
        self.BASELINES = baselines

        self.FIT_INTERCEPT = False


    def build_model_lab_meeting(self, hidden1, hidden2, hidden3):

        # create placeholders for input data
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self._phase_train = tf.placeholder(tf.bool, name='phase_train')
        self._combined_x = tf.placeholder(tf.float32, shape=[None, 4 * self.MIRLEN, 4 * self.SEQLEN, 1], name='biochem_x')
        self._pretrain_y =  tf.placeholder(tf.float32, shape=[None, 1], name='pretrain_y')

        # add layer 1
        with tf.name_scope('layer1'):
            self._w1, self._b1 = helpers.get_conv_params(4, 4, 1, hidden1, 'layer1')
            self._preactivate1 = tf.nn.conv2d(self._combined_x, self._w1, strides=[1, 4, 4, 1], padding='VALID') + self._b1
            self._preactivate1_bn = tf.layers.batch_normalization(self._preactivate1, axis=1, training=self._phase_train)
            self._layer1 = tf.nn.relu(self._preactivate1_bn)

        # add layer 2
        with tf.name_scope('layer2'):
            self._w2, self._b2 = helpers.get_conv_params(2, 2, hidden1, hidden2, 'layer2')
            self._preactivate2 = tf.nn.conv2d(self._layer1, self._w2, strides=[1, 1, 1, 1], padding='SAME') + self._b2
            self._preactivate2_bn = tf.layers.batch_normalization(self._preactivate2, axis=1, training=self._phase_train)
            self._layer2 = tf.nn.relu(self._preactivate2_bn)


        # add layer 3
        with tf.name_scope('layer3'):
            self._w3, self._b3 = helpers.get_conv_params(self.MIRLEN, self.SEQLEN, hidden2, hidden3, 'layer3')
            self._preactivate3 = tf.nn.conv2d(self._layer2, self._w3, strides=[1, self.MIRLEN, self.SEQLEN, 1], padding='VALID') + self._b3
            self._preactivate3_bn = tf.layers.batch_normalization(self._preactivate3, axis=1, training=self._phase_train)
            self._layer3 = tf.nn.relu(self._preactivate3_bn)

        # add dropout
        with tf.name_scope('dropout'):
            self._dropout = tf.nn.dropout(self._layer3, self._keep_prob)

        # reshape to 1D tensor
        self._layer_flat = tf.reshape(self._dropout, [-1, hidden3])

        # add last layer
        with tf.name_scope('final_layer'):
            with tf.name_scope('weights'):
                self._w4 = tf.get_variable("final_layer_weight", shape=[hidden3, 1],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                tf.add_to_collection('weight', self._w4)
            with tf.name_scope('biases'):
                self._b4 = tf.get_variable("final_layer_bias", shape=[1],
                                    initializer=tf.constant_initializer(0.0))
                tf.add_to_collection('bias', self._b4)

            # apply final layer
            self._pred_ind_values = (tf.matmul(self._layer_flat, self._w4) + self._b4)

        self._weight_size = tf.nn.l2_loss(self._w1) \
                            + tf.nn.l2_loss(self._w2) \
                            + tf.nn.l2_loss(self._w3) \
                            + tf.nn.l2_loss(self._w4)

        self._pretrain_loss = tf.nn.l2_loss(tf.subtract(self._pred_ind_values, self._pretrain_y))
        self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self._update_ops):
            self._train_step_pretrain = tf.train.AdamOptimizer(0.01).minimize(self._pretrain_loss)

        self.saver_pretrain = tf.train.Saver()

    def build_model(self, hidden1, hidden2, hidden3):

        # create placeholders for input data
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self._phase_train = tf.placeholder(tf.bool, name='phase_train')
        self._combined_x = tf.placeholder(tf.float32, shape=[None, 4 * self.MIRLEN, 4 * self.SEQLEN, 1], name='biochem_x')
        self._pretrain_y =  tf.placeholder(tf.float32, shape=[None, 1], name='pretrain_y')

        # add layer 1
        with tf.name_scope('layer1'):
            self._w1, self._b1 = helpers.get_conv_params(4, 4, 1, hidden1, 'layer1')
            self._preactivate1 = tf.nn.conv2d(self._combined_x, self._w1, strides=[1, 4, 4, 1], padding='VALID') + self._b1
            self._preactivate1_bn = tf.layers.batch_normalization(self._preactivate1, axis=1, training=self._phase_train)
            self._layer1 = tf.nn.relu(self._preactivate1_bn)

        # add layer 2
        with tf.name_scope('layer2'):
            self._w2, self._b2 = helpers.get_conv_params(2, 2, hidden1, hidden2, 'layer2')
            self._preactivate2 = tf.nn.conv2d(self._layer1, self._w2, strides=[1, 1, 1, 1], padding='VALID') + self._b2
            self._preactivate2_bn = tf.layers.batch_normalization(self._preactivate2, axis=1, training=self._phase_train)
            self._layer2 = tf.nn.relu(self._preactivate2_bn)

        # add layer 3
        with tf.name_scope('layer3'):
            self._w3, self._b3 = helpers.get_conv_params(4, 4, hidden2, hidden2, 'layer3')
            self._preactivate3 = tf.nn.conv2d(self._layer2, self._w3, strides=[1, 1, 1, 1], padding='VALID') + self._b3
            self._preactivate3_bn = tf.layers.batch_normalization(self._preactivate3, axis=1, training=self._phase_train)
            self._layer3 = tf.nn.relu(self._preactivate3_bn)

        # add layer 3
        with tf.name_scope('layer4'):
            self._w4, self._b4 = helpers.get_conv_params(self.MIRLEN-2, self.SEQLEN-2, hidden2, hidden3, 'layer4')
            self._preactivate4 = tf.nn.conv2d(self._layer2, self._w4, strides=[1, self.MIRLEN-2, self.SEQLEN-2, 1], padding='VALID') + self._b4
            self._preactivate4_bn = tf.layers.batch_normalization(self._preactivate4, axis=1, training=self._phase_train)
            self._layer4 = tf.nn.relu(self._preactivate4_bn)

        # add dropout
        with tf.name_scope('dropout'):
            self._dropout = tf.nn.dropout(self._layer4, self._keep_prob)

        # reshape to 1D tensor
        self._layer_flat = tf.reshape(self._dropout, [-1, hidden3])

        # add last layer
        with tf.name_scope('final_layer'):
            with tf.name_scope('weights'):
                self._w4 = tf.get_variable("final_layer_weight", shape=[hidden3, 1],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                tf.add_to_collection('weight', self._w4)
            with tf.name_scope('biases'):
                self._b4 = tf.get_variable("final_layer_bias", shape=[1],
                                    initializer=tf.constant_initializer(0.0))
                tf.add_to_collection('bias', self._b4)

            # apply final layer
            self._pred_ind_values = (tf.matmul(self._layer_flat, self._w4) + self._b4)

        self._weight_size = tf.nn.l2_loss(self._w1) \
                            + tf.nn.l2_loss(self._w2) \
                            + tf.nn.l2_loss(self._w3) \
                            + tf.nn.l2_loss(self._w4)

        self._pretrain_loss = tf.nn.l2_loss(tf.subtract(self._pred_ind_values, self._pretrain_y))
        self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self._update_ops):
            self._train_step_pretrain = tf.train.AdamOptimizer(0.01).minimize(self._pretrain_loss)

        self.saver_pretrain = tf.train.Saver()

    def build_model_all_conv(self, hidden1, hidden2):
        # create placeholders for input data
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self._phase_train = tf.placeholder(tf.bool, name='phase_train')
        self._combined_x = tf.placeholder(tf.float32, shape=[None, 4 * self.MIRLEN, 4 * self.SEQLEN, 1], name='biochem_x')
        self._pretrain_y =  tf.placeholder(tf.float32, shape=[None, 1], name='pretrain_y')

        # add layer 1 --> 20x12
        with tf.name_scope('layer1'):
            self._w1, self._b1 = helpers.get_conv_params(4, 4, 1, hidden1, 'layer1')
            self._preactivate1 = tf.nn.conv2d(self._combined_x, self._w1, strides=[1, 4, 4, 1], padding='VALID') + self._b1
            self._preactivate1_bn = tf.layers.batch_normalization(self._preactivate1, axis=1, training=self._phase_train)
            self._layer1 = tf.nn.relu(self._preactivate1_bn)

        # add layer 2 --> 19 x 11
        with tf.name_scope('layer2'):
            self._w2, self._b2 = helpers.get_conv_params(2, 2, hidden1, hidden2, 'layer2')
            self._preactivate2 = tf.nn.conv2d(self._layer1, self._w2, strides=[1, 1, 1, 1], padding='VALID') + self._b2
            self._preactivate2_bn = tf.layers.batch_normalization(self._preactivate2, axis=1, training=self._phase_train)
            self._layer2 = tf.nn.relu(self._preactivate2_bn)

        # add layer 3 --> 18 x 10
        with tf.name_scope('layer3'):
            self._w3, self._b3 = helpers.get_conv_params(2, 2, hidden2, hidden2, 'layer3')
            self._preactivate3 = tf.nn.conv2d(self._layer2, self._w3, strides=[1, 1, 1, 1], padding='VALID') + self._b3
            self._preactivate3_bn = tf.layers.batch_normalization(self._preactivate3, axis=1, training=self._phase_train)
            self._layer3 = tf.nn.relu(self._preactivate3_bn)

        # add layer 4 --> 17 x 9
        with tf.name_scope('layer4'):
            self._w4, self._b4 = helpers.get_conv_params(2, 2, hidden2, hidden2, 'layer4')
            self._preactivate4 = tf.nn.conv2d(self._layer3, self._w4, strides=[1, 1, 1, 1], padding='VALID') + self._b4
            self._preactivate4_bn = tf.layers.batch_normalization(self._preactivate4, axis=1, training=self._phase_train)
            self._layer4 = tf.nn.relu(self._preactivate4_bn)

        # add layer 5 --> 16 x 8
        with tf.name_scope('layer5'):
            self._w5, self._b5 = helpers.get_conv_params(2, 2, hidden2, hidden2, 'layer5')
            self._preactivate5 = tf.nn.conv2d(self._layer4, self._w5, strides=[1, 1, 1, 1], padding='VALID') + self._b5
            self._preactivate5_bn = tf.layers.batch_normalization(self._preactivate5, axis=1, training=self._phase_train)
            self._layer5 = tf.nn.relu(self._preactivate5_bn)

        # add layer 6 --> 15 x 7
        with tf.name_scope('layer6'):
            self._w6, self._b6 = helpers.get_conv_params(2, 2, hidden2, hidden2, 'layer6')
            self._preactivate6 = tf.nn.conv2d(self._layer5, self._w6, strides=[1, 1, 1, 1], padding='VALID') + self._b6
            self._preactivate6_bn = tf.layers.batch_normalization(self._preactivate6, axis=1, training=self._phase_train)
            self._layer6 = tf.nn.relu(self._preactivate6_bn)

        # add layer 7 --> 14 x 6
        with tf.name_scope('layer7'):
            self._w7, self._b7 = helpers.get_conv_params(2, 2, hidden2, hidden2, 'layer7')
            self._preactivate7 = tf.nn.conv2d(self._layer6, self._w7, strides=[1, 1, 1, 1], padding='VALID') + self._b7
            self._preactivate7_bn = tf.layers.batch_normalization(self._preactivate7, axis=1, training=self._phase_train)
            self._layer7 = tf.nn.relu(self._preactivate7_bn)

        # add layer 8 --> 13 x 5
        with tf.name_scope('layer8'):
            self._w8, self._b8 = helpers.get_conv_params(2, 2, hidden2, hidden2, 'layer8')
            self._preactivate8 = tf.nn.conv2d(self._layer7, self._w8, strides=[1, 1, 1, 1], padding='VALID') + self._b8
            self._preactivate8_bn = tf.layers.batch_normalization(self._preactivate8, axis=1, training=self._phase_train)
            self._layer8 = tf.nn.relu(self._preactivate8_bn)

        # add layer 9 --> 12 x 4
        with tf.name_scope('layer9'):
            self._w9, self._b9 = helpers.get_conv_params(2, 2, hidden2, hidden2, 'layer9')
            self._preactivate9 = tf.nn.conv2d(self._layer8, self._w9, strides=[1, 1, 1, 1], padding='VALID') + self._b9
            self._preactivate9_bn = tf.layers.batch_normalization(self._preactivate9, axis=1, training=self._phase_train)
            self._layer9 = tf.nn.relu(self._preactivate9_bn)

        # add layer 10 --> 11 x 3
        with tf.name_scope('layer10'):
            self._w10, self._b10 = helpers.get_conv_params(2, 2, hidden2, hidden2, 'layer10')
            self._preactivate10 = tf.nn.conv2d(self._layer9, self._w10, strides=[1, 1, 1, 1], padding='VALID') + self._b10
            self._preactivate10_bn = tf.layers.batch_normalization(self._preactivate10, axis=1, training=self._phase_train)
            self._layer10 = tf.nn.relu(self._preactivate10_bn)

        # add layer 11 --> 10 x 2
        with tf.name_scope('layer11'):
            self._w11, self._b11 = helpers.get_conv_params(2, 2, hidden2, hidden2, 'layer11')
            self._preactivate11 = tf.nn.conv2d(self._layer10, self._w11, strides=[1, 1, 1, 1], padding='VALID') + self._b11
            self._preactivate11_bn = tf.layers.batch_normalization(self._preactivate11, axis=1, training=self._phase_train)
            self._layer11 = tf.nn.relu(self._preactivate11_bn)

        # add layer 5 --> 9 x 1
        with tf.name_scope('layer12'):
            self._w12, self._b12 = helpers.get_conv_params(2, 2, hidden2, hidden2, 'layer12')
            self._preactivate12 = tf.nn.conv2d(self._layer11, self._w12, strides=[1, 1, 1, 1], padding='VALID') + self._b12
            self._preactivate12_bn = tf.layers.batch_normalization(self._preactivate12, axis=1, training=self._phase_train)
            self._layer12 = tf.nn.relu(self._preactivate12_bn)

        print(self._layer12)

        flat_dims = hidden2 * (self.MIRLEN - self.SEQLEN + 1)
        # reshape to 1D tensor
        self._layer_flat = tf.reshape(self._layer12, [-1, flat_dims])

        print(self._layer_flat)

        # add last layer
        with tf.name_scope('final_layer'):
            with tf.name_scope('weights'):
                self._wflat = tf.get_variable("final_layer_weight", shape=[flat_dims, 1],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                tf.add_to_collection('weight', self._wflat)
            with tf.name_scope('biases'):
                self._bflat = tf.get_variable("final_layer_bias", shape=[1],
                                    initializer=tf.constant_initializer(0.0))
                tf.add_to_collection('bias', self._bflat)

            # apply final layer
            self._pred_ind_values = (tf.matmul(self._layer_flat, self._wflat) + self._bflat)


        self._weight_size = tf.nn.l2_loss(self._w1) \
                            + tf.nn.l2_loss(self._w2) \
                            + tf.nn.l2_loss(self._w3) \
                            + tf.nn.l2_loss(self._w4) \
                            + tf.nn.l2_loss(self._w5) \
                            + tf.nn.l2_loss(self._w6) \
                            + tf.nn.l2_loss(self._w7) \
                            + tf.nn.l2_loss(self._w8) \
                            + tf.nn.l2_loss(self._w9) \
                            + tf.nn.l2_loss(self._w10) \
                            + tf.nn.l2_loss(self._w11) \
                            + tf.nn.l2_loss(self._w12) \
                            + tf.nn.l2_loss(self._wflat)


    # def add_repression_layers(self, batch_size_repression, batch_size_biochem, freeAGO_init):

    #     # make variables for real training
    #     self._repression_weight = tf.placeholder(tf.float32, name='repression_weight')
    #     self._biochem_y = tf.placeholder(tf.float32, shape=[None, 1], name='biochem_y')
    #     self._repression_max_size = tf.placeholder(tf.int32, shape=[], name='repression_max_size')
    #     self._repression_split_sizes = tf.placeholder(tf.int32, shape=[batch_size_repression * self.NUM_TRAIN * 2],
    #                                                     name='repression_split_sizes')
    #     self._expression_y = tf.placeholder(tf.float32, shape=[None, None], name='expression_y')
    #     self._intercept = tf.placeholder(tf.float32, shape=[None, 1], name='intercept')

    #     # construct global variables
    #     self._freeAGO_all = tf.get_variable('freeAGO_all', shape=[1, self.NUM_TRAIN * 2, 1],
    #                                         initializer=tf.constant_initializer(freeAGO_init))
    #     self._freeAGO_all_fixed = tf.get_variable('freeAGO_all_fixed', shape=[1, self.NUM_TRAIN * 2, 1],
    #                                         initializer=tf.constant_initializer(freeAGO_init), trainable=False)
    #     self._slope = tf.get_variable('slope', shape=(), initializer=tf.constant_initializer(-0.51023716))
    #     # self._decay = tf.get_variable('decay', shape=(), initializer=tf.constant_initializer(0.0))

    #     # construct a mask based on the number of sites per gene
    #     self._repression_mask = tf.reshape(tf.sequence_mask(self._repression_split_sizes, dtype=tf.float32),
    #                                   [batch_size_repression, self.NUM_TRAIN * 2, -1])

    #     # get padding dimensions
    #     self._repression_split_sizes_expand = tf.expand_dims(self._repression_split_sizes, 1)
    #     self._repression_paddings = tf.concat([tf.zeros(shape=tf.shape(self._repression_split_sizes_expand), dtype=tf.int32),
    #                                       self._repression_max_size - self._repression_split_sizes_expand], axis=1)
        
    #     # split data into biochem and repression
    #     self._pred_biochem = self._pred_ind_values[-1 * batch_size_biochem:, :]
    #     self._pred_repression_flat = tf.reshape(self._pred_ind_values[:-1 * batch_size_biochem, :], [-1])
    #     # self._pred_repression_flat = tf.reshape(self._pred_ind_values, [-1])

    #     # split repression data and pad into batch_size_biochem x self.NUM_TRAIN*2 x max_size matrix
    #     self._pred_repression_splits = tf.split(self._pred_repression_flat, self._repression_split_sizes)
    #     self._pred_repression_splits_padded = [tf.pad(self._pred_repression_splits[ix], self._repression_paddings[ix:ix+1,:]) for ix in range(batch_size_repression*self.NUM_TRAIN*2)]
    #     self._pred_repression_splits_padded_stacked = tf.stack(self._pred_repression_splits_padded)
    #     self._pred_repression = tf.reshape(self._pred_repression_splits_padded_stacked, [batch_size_repression, self.NUM_TRAIN*2, -1])


    #     # calculate predicted number bound and predicted log fold-change, without fitting freeAGO
    #     self._pred_nbound_split_fixed = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(self._freeAGO_all_fixed + self._pred_repression), self._repression_mask), axis=2)
    #     self._pred_nbound_fixed = tf.reduce_sum(tf.reshape(self._pred_nbound_split_fixed, [batch_size_repression, self.NUM_TRAIN, 2]), axis=2)

    #     # calculate predicted number bound and predicted log fold-change
    #     self._pred_nbound_split_fit = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(self._freeAGO_all + self._pred_repression), self._repression_mask), axis=2)
    #     self._pred_nbound_fit = tf.reduce_sum(tf.reshape(self._pred_nbound_split_fit, [batch_size_repression, self.NUM_TRAIN, 2]), axis=2)

    #     self._pred_logfc = (-0.5) * self._pred_nbound_fixed
    #     self._pred_logfc_fit = self._slope * self._pred_nbound_fit
    #     # self._pred_logfc = -1.0 * tf.log1p(self._pred_nbound / np.exp(0.0))
    #     # self._pred_logfc_fit = -1.0 * tf.log1p(self._pred_nbound / tf.exp(self._decay))

    #     self._weight_regularize = tf.multiply(self._weight_size, config.LAMBDA)

    #     self._intercept_fit = tf.reshape((tf.reduce_mean(self._expression_y, axis=1) - tf.reduce_mean(self._pred_logfc, axis=1)), [-1, 1])
    #     self._pred_tpm = self._pred_logfc + self._intercept
    #     self._pred_tpm_fit = self._pred_logfc_fit + self._intercept_fit

    #     self._biochem_loss = tf.nn.l2_loss(tf.subtract(self._pred_biochem, self._biochem_y)) / batch_size_biochem
    #     # self._repression_loss_weights = (tf.reduce_mean(tf.cast(tf.reshape(self._repression_split_sizes, [-1,2]), tf.float32), axis=1) + (2*tf.ones([batch_size_repression * self.NUM_TRAIN], dtype=tf.float32)))
    #     # self._repression_loss_weights = tf.reshape(self._repression_loss_weights, [batch_size_repression, self.NUM_TRAIN])
    #     # self._repression_loss = self._repression_weight * tf.nn.l2_loss(tf.subtract(self._pred_tpm, self._expression_y) / self._repression_loss_weights) / self.NUM_TRAIN
    #     # self._repression_loss_fit = self._repression_weight * tf.nn.l2_loss(tf.subtract(self._pred_tpm_fit, self._expression_y) / self._repression_loss_weights) / self.NUM_TRAIN

    #     self._repression_loss = self._repression_weight * tf.nn.l2_loss(tf.subtract(self._pred_tpm, self._expression_y)) / (self.NUM_TRAIN * batch_size_repression)
    #     self._repression_loss_fit = self._repression_weight * tf.nn.l2_loss(tf.subtract(self._pred_tpm_fit, self._expression_y)) / (self.NUM_TRAIN * batch_size_repression)

    #     self._loss = self._biochem_loss + self._repression_loss + self._weight_regularize
    #     self._loss_fit = self._biochem_loss + self._repression_loss_fit + self._weight_regularize

    #     self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     with tf.control_dependencies(self._update_ops):
    #         self._train_step = tf.train.AdamOptimizer(config.STARTING_LEARNING_RATE).minimize(self._loss)
    #         self._train_step_fit = tf.train.AdamOptimizer(config.STARTING_LEARNING_RATE).minimize(self._loss_fit)

    #     self.saver = tf.train.Saver(max_to_keep=config.NUM_EPOCHS)


    def add_repression_layers_mean_offset(self, batch_size_repression, batch_size_biochem, freeAGO_init, norm_ratio):

        # make variables for real training
        self._repression_weight = tf.placeholder(tf.float32, name='repression_weight')
        self._biochem_y = tf.placeholder(tf.float32, shape=[None, 1], name='biochem_y')
        self._repression_max_size = tf.placeholder(tf.int32, shape=[], name='repression_max_size')
        self._repression_split_sizes = tf.placeholder(tf.int32, shape=[batch_size_repression * self.NUM_TRAIN * 2],
                                                        name='repression_split_sizes')
        self._expression_y = tf.placeholder(tf.float32, shape=[None, None], name='expression_y')

        # construct global variables
        with tf.name_scope('train_later'):
            self._freeAGO_all = tf.get_variable('freeAGO_all', shape=[1, self.NUM_TRAIN * 2, 1],
                                                initializer=tf.constant_initializer(freeAGO_init))
            # self._slope = tf.get_variable('slope', shape=(), initializer=tf.constant_initializer(-0.51023716))
            self._decay = tf.get_variable('decay', shape=(), initializer=tf.constant_initializer(0.0))

        # construct a mask based on the number of sites per gene
        self._repression_mask = tf.reshape(tf.sequence_mask(self._repression_split_sizes, dtype=tf.float32),
                                      [batch_size_repression, self.NUM_TRAIN * 2, -1])

        # get padding dimensions
        self._repression_split_sizes_expand = tf.expand_dims(self._repression_split_sizes, 1)
        self._repression_paddings = tf.concat([tf.zeros(shape=tf.shape(self._repression_split_sizes_expand), dtype=tf.int32),
                                          self._repression_max_size - self._repression_split_sizes_expand], axis=1)
        
        # split data into biochem and repression
        self._pred_biochem = self._pred_ind_values[-1 * batch_size_biochem:, :]
        self._pred_repression_flat = tf.reshape(self._pred_ind_values[:-1 * batch_size_biochem, :], [-1]) * norm_ratio
        # self._pred_repression_flat = tf.reshape(self._pred_ind_values, [-1])

        # split repression data and pad into batch_size_biochem x self.NUM_TRAIN*2 x max_size matrix
        self._pred_repression_splits = tf.split(self._pred_repression_flat, self._repression_split_sizes)
        self._pred_repression_splits_padded = [tf.pad(self._pred_repression_splits[ix], self._repression_paddings[ix:ix+1,:]) for ix in range(batch_size_repression*self.NUM_TRAIN*2)]
        self._pred_repression_splits_padded_stacked = tf.stack(self._pred_repression_splits_padded)
        self._pred_repression = tf.reshape(self._pred_repression_splits_padded_stacked, [batch_size_repression, self.NUM_TRAIN*2, -1])

        # calculate predicted number bound and predicted log fold-change
        self._pred_nbound_split = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(self._freeAGO_all + self._pred_repression), self._repression_mask), axis=2)
        self._pred_nbound = tf.reduce_sum(tf.reshape(self._pred_nbound_split, [batch_size_repression, self.NUM_TRAIN, 2]), axis=2)

        # self._pred_logfc = self._slope * self._pred_nbound
        self._pred_logfc = -1.0 * tf.log1p(self._pred_nbound / tf.exp(self._decay))

        self._weight_regularize = tf.multiply(self._weight_size, config.LAMBDA)

        self._expression_mean = tf.reshape(tf.reduce_mean(self._expression_y, axis=1), [-1,1])
        self._pred_mean = tf.reshape(tf.reduce_mean(self._pred_logfc, axis=1), [-1,1])
        self._intercept_fit = self._expression_mean - self._pred_mean

        self._real_tpm = self._expression_y - self._expression_mean
        self._pred_tpm = self._pred_logfc - self._pred_mean

        self._biochem_loss = tf.nn.l2_loss(tf.subtract(self._pred_biochem, self._biochem_y)) / batch_size_biochem
        # self._repression_loss_weights = (tf.reduce_mean(tf.cast(tf.reshape(self._repression_split_sizes, [-1,2]), tf.float32), axis=1) + (2*tf.ones([batch_size_repression * self.NUM_TRAIN], dtype=tf.float32)))
        # self._repression_loss_weights = tf.reshape(self._repression_loss_weights, [batch_size_repression, self.NUM_TRAIN])
        # self._repression_loss = self._repression_weight * tf.nn.l2_loss(tf.subtract(self._pred_tpm, self._expression_y) / self._repression_loss_weights) / self.NUM_TRAIN
        # self._repression_loss_fit = self._repression_weight * tf.nn.l2_loss(tf.subtract(self._pred_tpm_fit, self._expression_y) / self._repression_loss_weights) / self.NUM_TRAIN

        self._repression_loss = self._repression_weight * tf.nn.l2_loss(tf.subtract(self._pred_tpm, self._real_tpm)) / (self.NUM_TRAIN * batch_size_repression)

        self._loss = self._biochem_loss + self._repression_loss + self._weight_regularize

        self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self._all_vars = tf.trainable_variables()
        self._most_vars = tf.trainable_variables()[:-2]

        with tf.control_dependencies(self._update_ops):
            self._train_step = tf.train.AdamOptimizer(config.STARTING_LEARNING_RATE).minimize(self._loss, var_list=self._most_vars)
            self._train_step_fit = tf.train.AdamOptimizer(config.STARTING_LEARNING_RATE).minimize(self._loss, var_list=self._all_vars)

        self.saver = tf.train.Saver(max_to_keep=config.NUM_EPOCHS)


    def initialize_vars(self):
        self.SESS.run(tf.global_variables_initializer())


    def plot_w1(self, filename):
        conv_weights = self.SESS.run(self._w1)
        xlabels = ['U','A','G','C']
        ylabels = ['A','U','C','G']
        helpers.graph_convolutions(conv_weights, xlabels, ylabels, filename)


    def plot_w3(self, filename):
        conv_weights = np.abs(self.SESS.run(self._w3))
        conv_weights = np.sum(conv_weights, axis=(2,3))
        vmin, vmax = np.min(conv_weights), np.max(conv_weights)
        xlabels = ['s{}'.format(i+1) for i in range(self.SEQLEN)]
        ylabels = ['m{}'.format(i+1) for i in list(range(self.MIRLEN))[::-1]]
        fig = plt.figure(figsize=(4,4))
        sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                    cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
        plt.savefig(filename)
        plt.close()


    def pretrain(self, pretrain_batch_x, pretrain_batch_y):
        
        # make feed_dict
        feed_dict = {
                        self._keep_prob: config.KEEP_PROB_TRAIN,
                        self._phase_train: True,
                        self._repression_weight: config.REPRESSION_WEIGHT,
                        self._combined_x: pretrain_batch_x,
                        self._pretrain_y: pretrain_batch_y
                    }

        _, l = self.self.SESS.run([self._train_step_pretrain, self._pretrain_loss], feed_dict=feed_dict)

        return l


    def save_pretrained_weights(self, filename):
        self.saver_pretrain.save(self.SESS, filename)


    def restore_pretrained_weights(self, path_to_saved_model):

        # restore previously pretrained weights
        latest = tf.train.latest_checkpoint(path_to_saved_model)
        print('Restoring from {}'.format(latest))
        self.saver_pretrain.restore(self.SESS, latest)

    def reset_final_layer(self):
        self.SESS.run(self._w3.initializer)
        self.SESS.run(self._b3.initializer)
        self.SESS.run(self._w4.initializer)
        self.SESS.run(self._b4.initializer)


    def train(self, batch_genes, batch_combined_x, batch_biochem_y, max_sites, train_sizes, batch_expression_y):
        if self.FIT_INTERCEPT:
            feed_dict = {
                            self._keep_prob: config.KEEP_PROB_TRAIN,
                            self._phase_train: True,
                            self._repression_weight: config.REPRESSION_WEIGHT,
                            self._combined_x: batch_combined_x,
                            self._biochem_y: batch_biochem_y,
                            self._repression_max_size: max_sites,
                            self._repression_split_sizes: train_sizes,
                            self._expression_y: batch_expression_y
                        }

            _, train_loss, b_loss, weight_reg, r_loss = self.SESS.run([self._train_step_fit,
                                                           self._loss,
                                                           self._biochem_loss,
                                                           self._weight_regularize,
                                                           self._repression_loss], feed_dict=feed_dict)

            # update gene baseline expression
            feed_dict = {
                            self._keep_prob: 1.0,
                            self._phase_train: False,
                            self._repression_weight: config.REPRESSION_WEIGHT,
                            self._combined_x: batch_combined_x,
                            self._biochem_y: batch_biochem_y,
                            self._repression_max_size: max_sites,
                            self._repression_split_sizes: train_sizes,
                            self._expression_y: batch_expression_y
                        }
            
            batch_intercept = self.SESS.run(self._intercept_fit, feed_dict=feed_dict)
            self.BASELINES.loc[batch_genes, 'nosite_tpm'] = batch_intercept.flatten()

        else:
            feed_dict = {
                            self._keep_prob: config.KEEP_PROB_TRAIN,
                            self._phase_train: True,
                            self._repression_weight: config.REPRESSION_WEIGHT,
                            self._combined_x: batch_combined_x,
                            self._biochem_y: batch_biochem_y,
                            self._repression_max_size: max_sites,
                            self._repression_split_sizes: train_sizes,
                            self._expression_y: batch_expression_y,
                            # self._intercept: self.BASELINES.loc[batch_genes][['nosite_tpm']].values
                        }

            _, train_loss, b_loss, weight_reg, r_loss = self.SESS.run([self._train_step,
                                                           self._loss,
                                                           self._biochem_loss,
                                                           self._weight_regularize,
                                                           self._repression_loss], feed_dict=feed_dict)

        return train_loss, b_loss, weight_reg, r_loss

    def predict_ind_values(self, x):

        feed_dict = {
                        self._keep_prob: 1.0,
                        self._phase_train: False,
                        self._combined_x: x
                    }

        return self.SESS.run(self._pred_ind_values, feed_dict=feed_dict)


    def predict_repression(self, x, max_sites, sizes):

        feed_dict = {
                        self._keep_prob: 1.0,
                        self._phase_train: False,
                        self._combined_x: x,
                        self._repression_max_size: max_sites,
                        self._repression_split_sizes: sizes
                    }

        # if self.FIT_INTERCEPT:
        #     return self.SESS.run(self._pred_logfc_fit, feed_dict=feed_dict)

        # else:
        #     return self.SESS.run(self._pred_logfc, feed_dict=feed_dict)


        return self.SESS.run(self._pred_logfc, feed_dict=feed_dict)

    def save(self, path, global_step):
        self.saver.save(self.SESS, path, global_step=global_step)




