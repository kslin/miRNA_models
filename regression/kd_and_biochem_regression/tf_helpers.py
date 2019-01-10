import numpy as np
import pandas as pd
import tensorflow as tf

import config, helpers, data_objects_endog

def make_ka_predictor(_combined_x, _pretrain_y, _keep_prob, _phase_train):
    # add layer 1
    with tf.name_scope('layer1'):
        _w1, _b1 = helpers.get_conv_params(4, 4, 1, config.HIDDEN1, 'layer1')
        _preactivate1 = tf.nn.conv2d(_combined_x, _w1, strides=[1, 4, 4, 1], padding='VALID') + _b1

        _preactivate1_bn = tf.contrib.layers.batch_norm(_preactivate1, is_training=_phase_train)

        _layer1 = tf.nn.leaky_relu(_preactivate1_bn)

    # add layer 2
    with tf.name_scope('layer2'):
        _w2, _b2 = helpers.get_conv_params(2, 2, config.HIDDEN1, config.HIDDEN2, 'layer2')
        _preactivate2 = tf.nn.conv2d(_layer1, _w2, strides=[1, 1, 1, 1], padding='VALID') + _b2

        _preactivate2_bn = tf.contrib.layers.batch_norm(_preactivate2, is_training=_phase_train)

        _layer2 = tf.nn.leaky_relu(_preactivate2_bn)

        _dropout2 = tf.nn.dropout(_layer2, _keep_prob)

    # add layer 3
    with tf.name_scope('layer3'):
        _w3, _b3 = helpers.get_conv_params(config.MIRLEN-1, config.SEQLEN-1, config.HIDDEN2, config.HIDDEN3, 'layer3')
        _preactivate3 = tf.nn.conv2d(_dropout2, _w3, strides=[1, config.MIRLEN-1, config.SEQLEN-1, 1], padding='VALID') + _b3

        _preactivate3_bn = tf.contrib.layers.batch_norm(_preactivate3, is_training=_phase_train)

        _layer3 = tf.nn.leaky_relu(_preactivate3_bn)

    # # add layer 2.5
    # with tf.name_scope('layer2_1'):
    #     _w2_1, _b2_1 = helpers.get_conv_params(2, 2, config.HIDDEN2, config.HIDDEN2, 'layer2_1')
    #     _preactivate2_1 = tf.nn.conv2d(_dropout2, _w2_1, strides=[1, 1, 1, 1], padding='VALID') + _b2_1

    #     _preactivate2_1_bn = tf.contrib.layers.batch_norm(_preactivate2_1, is_training=_phase_train)

    #     _layer2_1 = tf.nn.leaky_relu(_preactivate2_1_bn)

    #     _dropout2_1 = tf.nn.dropout(_layer2_1, _keep_prob)

    # # add layer 2.5
    # with tf.name_scope('layer2_2'):
    #     _w2_2, _b2_2 = helpers.get_conv_params(2, 2, config.HIDDEN2, config.HIDDEN2, 'layer2_2')
    #     _preactivate2_2 = tf.nn.conv2d(_dropout2_1, _w2_2, strides=[1, 1, 1, 1], padding='VALID') + _b2_2

    #     _preactivate2_2_bn = tf.contrib.layers.batch_norm(_preactivate2_2, is_training=_phase_train)

    #     _layer2_2 = tf.nn.leaky_relu(_preactivate2_2_bn)

    #     _dropout2_2 = tf.nn.dropout(_layer2_2, _keep_prob)

    # # add layer 3
    # with tf.name_scope('layer3'):
    #     _w3, _b3 = helpers.get_conv_params(config.MIRLEN-2, config.SEQLEN-2, config.HIDDEN2, config.HIDDEN3, 'layer3')
    #     _preactivate3 = tf.nn.conv2d(_dropout2_1, _w3, strides=[1, config.MIRLEN-2, config.SEQLEN-2, 1], padding='VALID') + _b3

    #     _preactivate3_bn = tf.contrib.layers.batch_norm(_preactivate3, is_training=_phase_train)

    #     _layer3 = tf.nn.leaky_relu(_preactivate3_bn)

    # add dropout
    with tf.name_scope('dropout'):
        _dropout = tf.nn.dropout(_layer3, _keep_prob)

    # reshape to 1D tensor
    _layer_flat = tf.reshape(_dropout, [-1, config.HIDDEN3])

    # add last layer
    with tf.name_scope('final_layer'):
        with tf.name_scope('weights'):
            _w4 = tf.get_variable("final_layer_weight", shape=[config.HIDDEN3, 1],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.add_to_collection('weight', _w4)
        with tf.name_scope('biases'):
            _b4 = tf.get_variable("final_layer_bias", shape=[1],
                                initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('bias', _b4)

        # apply final layer
        _norm_ratio = tf.constant(config.NORM_RATIO, name='norm_ratio')
        _pred_ind_values = tf.multiply(tf.add(tf.matmul(_layer_flat, _w4), _b4), _norm_ratio, name='pred_ka')

    _pretrain_loss = tf.nn.l2_loss(tf.subtract(_pred_ind_values, _pretrain_y))

    update_ops_pretrain = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops_pretrain):
        _pretrain_step = tf.train.AdamOptimizer(config.STARTING_LEARNING_RATE).minimize(_pretrain_loss)

    saver_pretrain = tf.train.Saver([_w1, _b1, _w2, _b2])

    return _pred_ind_values, _pretrain_loss, _pretrain_step, saver_pretrain, [_w1, _w2, _w3, _w4], [_b1, _b2, _b3, _b4]

# def reshape_data(num_train, batch_size_biochem, batch_size_repression, _pred_ind_values, _repression_max_size, _repression_split_sizes):
#     # construct a mask based on the number of sites per gene
#     _repression_mask = tf.reshape(tf.sequence_mask(_repression_split_sizes, dtype=tf.float32),
#                                   [batch_size_repression, num_train*2, -1])

#     # get padding dimensions
#     _repression_split_sizes_expand = tf.expand_dims(_repression_split_sizes, 1)
#     _repression_paddings = tf.concat([tf.zeros(shape=tf.shape(_repression_split_sizes_expand), dtype=tf.int32),
#                                       _repression_max_size - _repression_split_sizes_expand], axis=1)
    
#     # split data into biochem and repression
#     if batch_size_biochem == 0:
#         _pred_biochem = tf.constant(np.array([[0]]))
#         _pred_repression_flat = tf.reshape(_pred_ind_values, [-1])
#     else:
#         _pred_biochem = _pred_ind_values[-1 * batch_size_biochem:, :]
#         _pred_repression_flat = tf.reshape(_pred_ind_values[:-1 * batch_size_biochem, :], [-1])

#     # split repression data and pad into batch_size_biochem x num_train*2 x max_size matrix
#     _pred_repression_splits = tf.split(_pred_repression_flat, _repression_split_sizes)
#     _pred_repression_splits_padded = [tf.pad(_pred_repression_splits[ix], _repression_paddings[ix:ix+1,:]) for ix in range(batch_size_repression*num_train*2)]
#     _pred_repression_splits_padded_stacked = tf.stack(_pred_repression_splits_padded)
#     _pred_repression = tf.reshape(_pred_repression_splits_padded_stacked, [batch_size_repression, num_train*2, -1])

#     return _pred_biochem, _pred_repression_flat, _pred_repression, _repression_mask


# def test_reshape_data():
#     num_train = 2
#     batch_size_biochem = 2
#     batch_size_repression = 2
#     _pred_ind_values = tf.placeholder(tf.float32, shape=[None, 1])
#     _repression_max_size = tf.placeholder(tf.int32, shape=[])
#     _repression_split_sizes = tf.placeholder(tf.int32, shape=[batch_size_repression*num_train*2])

#     _pred_biochem, _pred_repression_flat, _pred_repression, _repression_mask = reshape_data(
#                                                                                         num_train,
#                                                                                         batch_size_biochem,
#                                                                                         batch_size_repression,
#                                                                                         _pred_ind_values,
#                                                                                         _repression_max_size,
#                                                                                         _repression_split_sizes
#                                                                                         )
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())

#         feed_dict = {
#             _pred_ind_values: np.array([1.0,2,1,3,2,2,3,4,1,1,1,2,3,2]).reshape([-1,1]),
#             _repression_max_size: 3,
#             _repression_split_sizes: np.array([1,0,1,1,2,3,3,1]), 

#         }

#         a, b, c, d = sess.run([_pred_biochem, _pred_repression_flat, _pred_repression, _repression_mask], feed_dict=feed_dict)

#         print(a)
#         print(b)
#         print(c)
#         print(d)


# def get_repression_loss(num_train, batch_size_repression, _pred_repression, _repression_mask, _let7_sites, _let7_mask, _utr_len, init_params):

#     _freeAGO_mean = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(init_params[0]))
#     _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset_toggle', shape=[num_train,1],
#                             initializer=tf.constant_initializer(init_params[1]))
#     _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset', shape=[num_train,1], initializer=tf.constant_initializer(init_params[2]))
#     _freeAGO_all = tf.reshape(tf.concat([_freeAGO_guide_offset + _freeAGO_mean, _freeAGO_pass_offset + _freeAGO_mean], axis=1), [1,num_train*2,1], name='freeAGO_all')

#     _decay = tf.get_variable('decay', shape=(), initializer=tf.constant_initializer(init_params[3]))

#     _utr_coef = tf.get_variable('utr_coef_toggle', shape=(),
#                                 initializer=tf.constant_initializer(init_params[4]))
#     _freeAGO_let7 = tf.get_variable('freeAGO_let7_toggle', shape=[1, 1, 1],
#                                     initializer=tf.constant_initializer(init_params[5]))

#     # calculate predicted number bound and predicted log fold-change
#     _pred_nbound_split = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO_all + _pred_repression), _repression_mask), axis=2)
#     _pred_nbound = tf.reduce_sum(tf.reshape(_pred_nbound_split, [batch_size_repression, num_train, 2]), axis=2)
#     _pred_nbound_let7 = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO_let7 + _let7_sites), _let7_mask), axis=2)
#     _pred_nbound_utr = tf.exp(_utr_coef) * _utr_len
#     _pred_nbound_init = _pred_nbound_let7 + _pred_nbound_utr
#     _pred_nbound_total = _pred_nbound + _pred_nbound_init

#     _pred_logfc_init = tf.multiply(tf.log1p(_pred_nbound_init / tf.exp(_decay)), -1.0, name='pred_logfc_init')
#     _pred_logfc = tf.multiply(tf.log1p(_pred_nbound_total / tf.exp(_decay)), -1.0, name='pred_logfc')
#     _pred_logfc_net = tf.subtract(_pred_logfc, _pred_logfc_init, name='pred_logfc_net')

#     return _freeAGO_mean, _freeAGO_all, _decay, _utr_coef, _freeAGO_let7, _pred_logfc, _pred_logfc_net

def both_steps_simple(scope, num_train, batch_size_biochem, batch_size_repression, _pred_ind_values,
                _repression_max_size, _repression_split_sizes, _utr_len, init_params):

    # construct a mask based on the number of sites per gene
    # _repression_mask = tf.reshape(tf.sequence_mask(_repression_split_sizes, dtype=tf.float32),
    #                               [batch_size_repression, num_train*2, -1])

    # get padding dimensions
    _repression_split_sizes_expand = tf.expand_dims(_repression_split_sizes, 1)
    _repression_paddings = tf.concat([tf.zeros(shape=tf.shape(_repression_split_sizes_expand), dtype=tf.int32),
                                      _repression_max_size - _repression_split_sizes_expand], axis=1)
    
    # split data into biochem and repression
    if batch_size_biochem == 0:
        _pred_biochem = tf.constant(np.array([[0]]))
        _pred_repression_flat = tf.reshape(_pred_ind_values, [-1])
    else:
        _pred_biochem = _pred_ind_values[-1 * batch_size_biochem:, :]
        _pred_repression_flat = tf.reshape(_pred_ind_values[:-1 * batch_size_biochem, :], [-1])

    # split repression data and pad into batch_size_biochem x num_train*2 x max_size matrix
    _pred_repression_splits = tf.split(_pred_repression_flat, _repression_split_sizes)
    _pred_repression_splits_padded = [tf.pad(_pred_repression_splits[ix], _repression_paddings[ix:ix+1,:]) for ix in range(batch_size_repression*num_train*2)]
    _pred_repression_splits_padded_stacked = tf.stack(_pred_repression_splits_padded)
    _pred_repression = tf.reshape(_pred_repression_splits_padded_stacked, [batch_size_repression, num_train*2, -1])

    _freeAGO_mean = tf.get_variable('freeAGO_mean_{}'.format(scope), shape=(), initializer=tf.constant_initializer(init_params[0]))
    _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset_toggle_{}'.format(scope), shape=[num_train,1],
                            initializer=tf.constant_initializer(init_params[1]))
    _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset_{}'.format(scope), shape=[num_train,1], initializer=tf.constant_initializer(init_params[2]))
    _freeAGO_all = tf.reshape(tf.concat([_freeAGO_guide_offset + _freeAGO_mean, _freeAGO_pass_offset + _freeAGO_mean], axis=1), [1,num_train*2,1], name='freeAGO_all_{}'.format(scope))

    # _freeAGO_mean = tf.get_variable('freeAGO_mean_{}'.format(scope), shape=(), initializer=tf.constant_initializer(init_params[0]))
    # _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset_toggle_{}'.format(scope), shape=[num_train,1],
    #                         initializer=tf.constant_initializer(init_params[0] + init_params[1]))
    # _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset_{}'.format(scope), shape=[num_train,1], initializer=tf.constant_initializer(init_params[0] + init_params[2]))
    # _freeAGO_all = tf.reshape(tf.concat([_freeAGO_guide_offset, _freeAGO_pass_offset], axis=1), [1,num_train*2,1], name='freeAGO_all_{}'.format(scope))

    _decay = tf.get_variable('decay_{}'.format(scope), shape=(), initializer=tf.constant_initializer(init_params[3]))

    _utr_coef = tf.get_variable('utr_coef_toggle_{}'.format(scope), shape=(),
                                initializer=tf.constant_initializer(init_params[4]))

    _pred_nbound_utr = tf.exp(_utr_coef) * _utr_len
    
    # _utr_slope = 1.0 / _utr_len * tf.exp(_utr_coef) + tf.exp(_decay)
    # _utr_slope = tf.nn.relu(_decay + (tf.log(_utr_len) * _utr_coef))

    _repression_mask = tf.cast(_pred_repression > 0, tf.float32)

    # calculate predicted number bound and predicted log fold-change
    _pred_nbound_split = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO_all + _pred_repression), _repression_mask), axis=2)
    _pred_nbound = tf.reduce_sum(tf.reshape(_pred_nbound_split, [batch_size_repression, num_train, 2]), axis=2)

    _pred_logfc_init = tf.multiply(tf.log1p(_pred_nbound_utr * tf.exp(_decay)), -1.0, name='pred_logfc_init_{}'.format(scope))
    _pred_logfc = tf.multiply(tf.log1p((_pred_nbound + _pred_nbound_utr) * tf.exp(_decay)), -1.0, name='pred_logfc_{}'.format(scope))
    _pred_logfc_net = tf.subtract(_pred_logfc, _pred_logfc_init, name='pred_logfc_net_{}'.format(scope))

    return _pred_biochem, _pred_repression_flat, _freeAGO_mean, _freeAGO_guide_offset, _freeAGO_pass_offset, _freeAGO_all, _decay, _utr_coef, _pred_logfc

# def both_steps(scope, num_train, batch_size_biochem, batch_size_repression, _pred_ind_values,
#                 _repression_max_size, _repression_split_sizes, _let7_sites, _let7_mask, _utr_len, init_params):

#     # construct a mask based on the number of sites per gene
#     _repression_mask = tf.reshape(tf.sequence_mask(_repression_split_sizes, dtype=tf.float32),
#                                   [batch_size_repression, num_train*2, -1])

#     # get padding dimensions
#     _repression_split_sizes_expand = tf.expand_dims(_repression_split_sizes, 1)
#     _repression_paddings = tf.concat([tf.zeros(shape=tf.shape(_repression_split_sizes_expand), dtype=tf.int32),
#                                       _repression_max_size - _repression_split_sizes_expand], axis=1)
    
#     # split data into biochem and repression
#     if batch_size_biochem == 0:
#         _pred_biochem = tf.constant(np.array([[0]]))
#         _pred_repression_flat = tf.reshape(_pred_ind_values, [-1])
#     else:
#         _pred_biochem = _pred_ind_values[-1 * batch_size_biochem:, :]
#         _pred_repression_flat = tf.reshape(_pred_ind_values[:-1 * batch_size_biochem, :], [-1])

#     # split repression data and pad into batch_size_biochem x num_train*2 x max_size matrix
#     _pred_repression_splits = tf.split(_pred_repression_flat, _repression_split_sizes)
#     _pred_repression_splits_padded = [tf.pad(_pred_repression_splits[ix], _repression_paddings[ix:ix+1,:]) for ix in range(batch_size_repression*num_train*2)]
#     _pred_repression_splits_padded_stacked = tf.stack(_pred_repression_splits_padded)
#     _pred_repression = tf.reshape(_pred_repression_splits_padded_stacked, [batch_size_repression, num_train*2, -1])

#     _freeAGO_mean = tf.get_variable('freeAGO_mean_{}'.format(scope), shape=(), initializer=tf.constant_initializer(init_params[0]))
#     _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset_toggle_{}'.format(scope), shape=[num_train,1],
#                             initializer=tf.constant_initializer(init_params[1]))
#     _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset_{}'.format(scope), shape=[num_train,1], initializer=tf.constant_initializer(init_params[2]))
#     _freeAGO_all = tf.reshape(tf.concat([_freeAGO_guide_offset + _freeAGO_mean, _freeAGO_pass_offset + _freeAGO_mean], axis=1), [1,num_train*2,1], name='freeAGO_all_{}'.format(scope))

#     _decay = tf.get_variable('decay_{}'.format(scope), shape=(), initializer=tf.constant_initializer(init_params[3]))

#     _utr_coef = tf.get_variable('utr_coef_toggle_{}'.format(scope), shape=(),
#                                 initializer=tf.constant_initializer(init_params[4]))
#     # _freeAGO_let7 = tf.get_variable('freeAGO_let7_toggle_{}'.format(scope), shape=[1, 1, 1],
#     #                                 initializer=tf.constant_initializer(init_params[5]))

#     # calculate predicted number bound and predicted log fold-change
#     _pred_nbound_split = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO_all + _pred_repression), _repression_mask), axis=2)
#     _pred_nbound = tf.reduce_sum(tf.reshape(_pred_nbound_split, [batch_size_repression, num_train, 2]), axis=2)
#     _pred_nbound_let7 = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO_let7 + _let7_sites), _let7_mask), axis=2)
#     _pred_nbound_utr = tf.exp(_utr_coef) * _utr_len
#     _pred_nbound_init = _pred_nbound_let7 + _pred_nbound_utr
#     _pred_nbound_total = _pred_nbound + _pred_nbound_init

#     _pred_logfc_init = tf.multiply(tf.log1p(_pred_nbound_init / tf.exp(_decay)), -1.0, name='pred_logfc_init_{}'.format(scope))
#     _pred_logfc = tf.multiply(tf.log1p(_pred_nbound_total / tf.exp(_decay)), -1.0, name='pred_logfc_{}'.format(scope))
#     _pred_logfc_net = tf.subtract(_pred_logfc, _pred_logfc_init, name='pred_logfc_net_{}'.format(scope))

#     return _pred_biochem, _pred_repression_flat, _freeAGO_mean, _freeAGO_guide_offset, _freeAGO_all, _decay, _utr_coef, _freeAGO_let7, _pred_logfc, _pred_logfc_net



# def test_get_repression_loss():
#     num_train = 2
#     batch_size_biochem = 2
#     batch_size_repression = 2
#     _pred_ind_values = tf.placeholder(tf.float32, shape=[None, 1])
#     _repression_max_size = tf.placeholder(tf.int32, shape=[])
#     _repression_split_sizes = tf.placeholder(tf.int32, shape=[batch_size_repression*num_train*2])
#     _utr_len = tf.placeholder(tf.float32, shape=[None, 1])
#     _let7_sites = tf.placeholder(tf.float32, shape=[None, 1, None])
#     _let7_mask = tf.placeholder(tf.float32, shape=[None, 1, None])

#     _pred_biochem, _pred_repression_flat, _pred_repression, _repression_mask = reshape_data(
#                                                                                         num_train,
#                                                                                         batch_size_biochem,
#                                                                                         batch_size_repression,
#                                                                                         _pred_ind_values,
#                                                                                         _repression_max_size,
#                                                                                         _repression_split_sizes
#                                                                                         )

#     # freeAGO_mean, guide_offset, pass_offset, decay, utr_coef, let7_freeAGO
#     init_params = [-2.0, 0.0, -1.0, 0.0, -9.0, -8.0]
#     _freeAGO_mean, _freeAGO_all, _decay, _utr_coef, _freeAGO_let7, _pred_logfc, pred_logfc_net = get_repression_loss(
#                                                                                         num_train,
#                                                                                         batch_size_repression,
#                                                                                         _pred_repression,
#                                                                                         _repression_mask,
#                                                                                         _let7_sites,
#                                                                                         _let7_mask,
#                                                                                         _utr_len,
#                                                                                         init_params
#                                                                                     )
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())

#         feed_dict = {
#             _pred_ind_values: np.array([1.0,2,1,3,2,2,3,4,1,1,1,2,3,2]).reshape([-1,1]),
#             _repression_max_size: 3,
#             _repression_split_sizes: np.array([1,0,1,1,2,3,3,1]),
#             _utr_len: np.array([2000, 3000]).reshape([-1,1]),
#             _let7_sites: np.array([[4.0, 5, 6],[7,0,0]]).reshape([2,1,-1]),
#             _let7_mask: np.array([[1.0, 1.0, 1.0],[1.0,0,0]]).reshape([2,1,-1]),
#         }

#         blah = sess.run(_pred_logfc, feed_dict=feed_dict)

#         actual = np.array([[-0.53084946, -0.71815073],
#                     [-1.4748155, -0.99876857]])

#         print(blah)

#         print(np.sum(np.abs(blah - actual)))

# def test_both_steps():
#     num_train = 2
#     batch_size_biochem = 2
#     batch_size_repression = 2
#     _pred_ind_values = tf.placeholder(tf.float32, shape=[None, 1])
#     _repression_max_size = tf.placeholder(tf.int32, shape=[])
#     _repression_split_sizes = tf.placeholder(tf.int32, shape=[batch_size_repression*num_train*2])
#     _utr_len = tf.placeholder(tf.float32, shape=[None, 1])
#     _let7_sites = tf.placeholder(tf.float32, shape=[None, 1, None])
#     _let7_mask = tf.placeholder(tf.float32, shape=[None, 1, None])
#     init_params = [-2.0, 0.0, -1.0, 0.0, -9.0, -8.0]

#     results = both_steps('blah', num_train, batch_size_biochem, batch_size_repression, _pred_ind_values,
#                 _repression_max_size, _repression_split_sizes, _let7_sites, _let7_mask, _utr_len, init_params)
#     _pred_biochem, _pred_repression_flat, _freeAGO_mean, _freeAGO_all, _decay, _utr_coef, _freeAGO_let7, _pred_logfc, _pred_logfc_net = results

#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())

#         feed_dict = {
#             _pred_ind_values: np.array([1.0,2,1,3,2,2,3,4,1,1,1,2,3,2]).reshape([-1,1]),
#             _repression_max_size: 3,
#             _repression_split_sizes: np.array([1,0,1,1,2,3,3,1]),
#             _utr_len: np.array([2000, 3000]).reshape([-1,1]),
#             _let7_sites: np.array([[4.0, 5, 6],[7,0,0]]).reshape([2,1,-1]),
#             _let7_mask: np.array([[1.0, 1.0, 1.0],[1.0,0,0]]).reshape([2,1,-1]),
#         }

#         blah = sess.run(_pred_logfc, feed_dict=feed_dict)

#         actual = np.array([[-0.53084946, -0.71815073],
#                     [-1.4748155, -0.99876857]])

#         print(blah)

#         print(np.sum(np.abs(blah - actual)))

# test_both_steps()   



    


