import tensorflow as tf

import config


def get_conv_params(dim1, dim2, in_channels, out_channels, layer_name):

    # create variables for weights and biases
    with tf.name_scope('weights'):
        weights = tf.get_variable("{}_weight".format(layer_name),
                               shape=[dim1, dim2, in_channels, out_channels],
                               initializer=tf.truncated_normal_initializer(stddev=0.1))

        # add variable to collection of variables
        tf.add_to_collection('weight', weights)
    with tf.name_scope('biases'):
        biases = tf.get_variable("{}_bias".format(layer_name), shape=[out_channels],
                              initializer=tf.constant_initializer(0.0))

        # add variable to collection of variables
        tf.add_to_collection('bias', biases)

    return weights, biases


def seq2ka_predictor(_combined_x, _keep_prob, _phase_train):
    # add layer 1
    with tf.name_scope('layer1'):
        _w1, _b1 = get_conv_params(4, 4, 1, config.HIDDEN1, 'layer1')
        _preactivate1 = tf.nn.conv2d(_combined_x, _w1, strides=[1, 4, 4, 1], padding='VALID') + _b1

        _preactivate1_bn = tf.contrib.layers.batch_norm(_preactivate1, is_training=_phase_train)

        _layer1 = tf.nn.leaky_relu(_preactivate1_bn)

    # add layer 2
    with tf.name_scope('layer2'):
        _w2, _b2 = get_conv_params(2, 2, config.HIDDEN1, config.HIDDEN2, 'layer2')
        _preactivate2 = tf.nn.conv2d(_layer1, _w2, strides=[1, 1, 1, 1], padding='VALID') + _b2

        _preactivate2_bn = tf.contrib.layers.batch_norm(_preactivate2, is_training=_phase_train)

        _layer2 = tf.nn.leaky_relu(_preactivate2_bn)

        _dropout2 = tf.nn.dropout(_layer2, _keep_prob)

    # add layer 3
    with tf.name_scope('layer3'):
        _w3, _b3 = get_conv_params(config.MIRLEN - 1, config.SEQLEN - 1, config.HIDDEN2, config.HIDDEN3, 'layer3')
        _preactivate3 = tf.nn.conv2d(_dropout2, _w3, strides=[1, 1, 1, 1], padding='VALID') + _b3

        _preactivate3_bn = tf.contrib.layers.batch_norm(_preactivate3, is_training=_phase_train)

        _layer3 = tf.nn.leaky_relu(_preactivate3_bn)

    print('layer1: {}'.format(_layer1))
    print('layer2: {}'.format(_layer2))
    print('layer3: {}'.format(_layer3))

    # add dropout
    with tf.name_scope('dropout'):
        _dropout = tf.nn.dropout(_layer3, _keep_prob)

    # _max_pool = tf.layers.max_pooling2d(
    #     _dropout,
    #     (1, config.SEQLEN - 1),
    #     1,
    #     padding='VALID',
    #     data_format='channels_last',
    #     name='max_pool'
    # )
    # print('max_pool: {}'.format(_max_pool))

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
        _pred_ka_values = tf.nn.relu(tf.add(tf.matmul(_layer_flat, _w4), _b4), name='pred_ka')

    print('pred_ka: {}'.format(_pred_ka_values))

    _cnn_weights = {
        'w1': _w1,
        'b1': _b1,
        'w2': _w2,
        'b2': _b2,
        'w3': _w3,
        'b3': _b3,
        'w4': _w4,
        'b4': _b4,
    }

    return _pred_ka_values, _cnn_weights


def ka_ts7_to_repression(_utr_ka_values, _ts7_feats, _freeAGO_concs, _feat_weights):
    _nbound = tf.sigmoid(_utr_ka_values + _freeAGO_concs)
    


def pad_kd_from_genes(_utr_ka_values, _utr_split_sizes, _utr_max_size, num_train, batch_size_repression, passenger):
    if passenger:
        total_num_train = 2 * num_train
    else:
        total_num_train = num_train

    # get padding dimensions
    _utr_split_sizes_expand = tf.expand_dims(_utr_split_sizes, 1)
    _utr_paddings = tf.concat([tf.zeros(shape=tf.shape(_utr_split_sizes_expand), dtype=tf.int32),
                                      tf.reduce_max(_utr_split_sizes) - _utr_split_sizes_expand], axis=1)

    # split repression data and pad into batch_size_biochem x num_train*2 x max_size matrix
    _pred_utr_splits = tf.split(_utr_ka_values, _utr_split_sizes)
    _pred_utr_splits_padded = [tf.pad(_pred_utr_splits[ix], _utr_paddings[ix:ix + 1, :]) for ix in range(batch_size_repression * total_num_train)]
    _pred_utr_splits_padded_stacked = tf.stack(_pred_utr_splits_padded)
    _utr_ka_values_reshaped = tf.reshape(_pred_utr_splits_padded_stacked, [batch_size_repression, total_num_train, -1])

    return _utr_ka_values_reshaped


def ka2repression_predictor(scope, _utr_ka_values_reshaped, _utr_len, num_train, batch_size_repression, init_params):

    _freeAGO_mean = tf.get_variable('freeAGO_mean_{}'.format(scope), shape=(), initializer=tf.constant_initializer(init_params[0]))
    _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset_{}'.format(scope), shape=[num_train, 1],
                            initializer=tf.constant_initializer(init_params[1]))
    _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset_{}'.format(scope), shape=[num_train, 1], initializer=tf.constant_initializer(init_params[2]))
    _freeAGO_all = tf.reshape(tf.concat([_freeAGO_guide_offset + _freeAGO_mean, _freeAGO_pass_offset + _freeAGO_mean], axis=1), [1, num_train * 2, 1], name='freeAGO_all_{}'.format(scope))

    _decay = tf.get_variable('decay_{}'.format(scope), shape=(), initializer=tf.constant_initializer(init_params[3]))

    _utr_coef = tf.get_variable('utr_coef_{}'.format(scope), shape=(),
                                initializer=tf.constant_initializer(init_params[4]))

    _pred_nbound_utr = tf.exp(_utr_coef) * _utr_len

    _utr_mask = tf.cast(_utr_ka_values_reshaped > 0, tf.float32)

    # calculate predicted number bound and predicted log fold-change
    _pred_nbound_split = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO_all + _utr_ka_values_reshaped), _utr_mask), axis=2)
    _pred_nbound = tf.reduce_sum(tf.reshape(_pred_nbound_split, [batch_size_repression, num_train, 2]), axis=2)

    _pred_logfc_init = tf.multiply(tf.log1p(_pred_nbound_utr * tf.exp(_decay)), -1.0, name='pred_logfc_init_{}'.format(scope))
    _pred_logfc = tf.multiply(tf.log1p((_pred_nbound + _pred_nbound_utr) * tf.exp(_decay)), -1.0, name='pred_logfc_{}'.format(scope))
    _pred_logfc_net = tf.subtract(_pred_logfc, _pred_logfc_init, name='pred_logfc_net_{}'.format(scope))

    _results = {
        'freeAGO_mean': _freeAGO_mean,
        'freeAGO_guide_offset': _freeAGO_guide_offset,
        'freeAGO_pass_offset': _freeAGO_pass_offset,
        'freeAGO_all': _freeAGO_all,
        'decay': _decay,
        'utr_coef': _utr_coef,
        'pred_logfc': _pred_logfc,
        'pred_logfc_net': _pred_logfc_net
    }

    return _results

# def ka2featuremat(_utr_ka_values_reshaped, _ts7_features, num_train, batch_size_repression, ):
#     _freeAGO_mean = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(config.FREEAGO_INIT))
#     _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset', shape=[num_train, 1],
#                             initializer=tf.constant_initializer(config.GUIDE_OFFSET_INIT))
#     _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset', shape=[num_train, 1],
#                             initializer=tf.constant_initializer(config.PASS_OFFSET_INIT))
#     _freeAGO_all = tf.reshape(tf.concat([_freeAGO_guide_offset + _freeAGO_mean, _freeAGO_pass_offset + _freeAGO_mean], axis=1),
#                              [1, num_train * 2, 1], name='freeAGO_all')

#     _pred_nbound_split = tf.nn.sigmoid(_freeAGO_all + _utr_ka_values_reshaped)
#     _pred_features_split = tf.concat([tf.expand_dims(_pred_nbound_split, axis=3), _ts7_features], axis=3)
#     _utr_mask = tf.cast(_utr_ka_values_reshaped > 0, tf.float32)

#     # add last layer
#     with tf.name_scope('ts7_layer'):
#         with tf.name_scope('weights'):
#             _w5 = tf.get_variable("ts7_layer_weight", shape=[config.NUM_TS7 + 1, 1, 1],
#                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
#             tf.add_to_collection('weight', _w5)

#         _pred_logfc_split = tf.reduce_sum(tf.multiply(tf.matmul(_pred_features_split, _w5), _utr_mask), axis=2, name='pred_ka')
#         _pred_logfc = tf.reduce_sum(tf.reshape(_pred_logfc_split, [batch_size_repression, num_train, 2]), axis=2)

#     _results = {
#         'freeAGO_guide_offset': _freeAGO_guide_offset,
#         'freeAGO_pass_offset': _freeAGO_pass_offset,
#         'freeAGO_all': _freeAGO_all,
#         'w5': _w5,
#         'pred_logfc': _pred_logfc
#     }

#     return _results



# def test_get_repression_loss():
#     num_train = 2
#     batch_size_biochem = 2
#     batch_size_repression = 2
#     _pred_ka_values = tf.placeholder(tf.float32, shape=[None, 1])
#     _repression_max_size = tf.placeholder(tf.int32, shape=[])
#     _repression_split_sizes = tf.placeholder(tf.int32, shape=[batch_size_repression*num_train*2])
#     _utr_len = tf.placeholder(tf.float32, shape=[None, 1])
#     _let7_sites = tf.placeholder(tf.float32, shape=[None, 1, None])
#     _let7_mask = tf.placeholder(tf.float32, shape=[None, 1, None])

#     _pred_biochem, _pred_repression_flat, _pred_repression, _repression_mask = reshape_data(
#                                                                                         num_train,
#                                                                                         batch_size_biochem,
#                                                                                         batch_size_repression,
#                                                                                         _pred_ka_values,
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
#             _pred_ka_values: np.array([1.0,2,1,3,2,2,3,4,1,1,1,2,3,2]).reshape([-1,1]),
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
#     _pred_ka_values = tf.placeholder(tf.float32, shape=[None, 1])
#     _repression_max_size = tf.placeholder(tf.int32, shape=[])
#     _repression_split_sizes = tf.placeholder(tf.int32, shape=[batch_size_repression*num_train*2])
#     _utr_len = tf.placeholder(tf.float32, shape=[None, 1])
#     _let7_sites = tf.placeholder(tf.float32, shape=[None, 1, None])
#     _let7_mask = tf.placeholder(tf.float32, shape=[None, 1, None])
#     init_params = [-2.0, 0.0, -1.0, 0.0, -9.0, -8.0]

#     results = both_steps('blah', num_train, batch_size_biochem, batch_size_repression, _pred_ka_values,
#                 _repression_max_size, _repression_split_sizes, _let7_sites, _let7_mask, _utr_len, init_params)
#     _pred_biochem, _pred_repression_flat, _freeAGO_mean, _freeAGO_all, _decay, _utr_coef, _freeAGO_let7, _pred_logfc, _pred_logfc_net = results

#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())

#         feed_dict = {
#             _pred_ka_values: np.array([1.0,2,1,3,2,2,3,4,1,1,1,2,3,2]).reshape([-1,1]),
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
