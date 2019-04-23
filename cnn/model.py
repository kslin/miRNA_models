import numpy as np
import pandas as pd
import tensorflow as tf

import parse_data_utils

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def get_conv_params(dim1, dim2, in_channels, out_channels, layer_name):

    # create variables for weights and biases
    with tf.name_scope('weights'):
        weights = tf.get_variable("{}_weight".format(layer_name),
                               shape=[dim1, dim2, in_channels, out_channels],
                               initializer=tf.contrib.layers.xavier_initializer())
                               # initializer=tf.truncated_normal_initializer(stddev=0.05))

        # add variable to collection of variables
        tf.add_to_collection('weight', weights)
        variable_summaries(weights)

    return weights

def seq2ka_predictor(input_data, dropout_rate, phase_train, hidden1, hidden2, hidden3, mirlen, seqlen):
    # add layer 1
    with tf.name_scope('layer1'):
        _w1 = get_conv_params(4, 4, 1, hidden1, 'layer1')
        _preactivate1 = tf.nn.conv2d(input_data, _w1, strides=[1, 4, 4, 1], padding='VALID')
        # _preactivate1_bn = tf.keras.layers.BatchNormalization(axis=-1, scale=False)(_preactivate1, training=phase_train)
        _preactivate1_bn = tf.layers.batch_normalization(_preactivate1, training=phase_train, renorm=True)
        _layer1 = tf.nn.leaky_relu(_preactivate1_bn)
        tf.summary.histogram('activations', _layer1)
        # _dropout1 = tf.nn.dropout(_layer1, rate=dropout_rate)

    # add layer 2
    with tf.name_scope('layer2'):
        _w2 = get_conv_params(2, 2, hidden1, hidden2, 'layer2')
        _preactivate2 = tf.nn.conv2d(_layer1, _w2, strides=[1, 1, 1, 1], padding='VALID')
        # _preactivate2_bn = tf.keras.layers.BatchNormalization(axis=-1, scale=False)(_preactivate2, training=phase_train)
        _preactivate2_bn = tf.layers.batch_normalization(_preactivate2, training=phase_train, renorm=True)
        _layer2 = tf.nn.leaky_relu(_preactivate2_bn)
        tf.summary.histogram('activations', _layer2)

        _dropout2 = tf.nn.dropout(_layer2, rate=(dropout_rate/2))

    # add layer 2.5
    with tf.name_scope('layer2_5'):
        _w2_5 = get_conv_params(4, 4, hidden2, hidden2, 'layer2_5')
        _preactivate2_5 = tf.nn.conv2d(_dropout2, _w2_5, strides=[1, 1, 1, 1], padding='VALID')
        # _preactivate2_5_bn = tf.keras.layers.BatchNormalization(axis=-1, scale=False)(_preactivate2_5, training=phase_train)
        _preactivate2_5_bn = tf.layers.batch_normalization(_preactivate2_5, training=phase_train, renorm=True)
        _layer2_5 = tf.nn.leaky_relu(_preactivate2_5_bn)
        tf.summary.histogram('activations', _layer2_5)
        _dropout2_5 = tf.nn.dropout(_layer2_5, rate=dropout_rate/2)

    # add layer 3
    with tf.name_scope('layer3'):
        _w3 = get_conv_params(mirlen - 4, seqlen - 4, hidden2, hidden3, 'layer3')
        _preactivate3 = tf.nn.conv2d(_dropout2_5, _w3, strides=[1, 1, 1, 1], padding='VALID')
        # _preactivate3_bn = tf.keras.layers.BatchNormalization()(_preactivate3, training=phase_train)
        _preactivate3_bn = tf.layers.batch_normalization(_preactivate3, training=phase_train, renorm=True)
        _layer3 = tf.nn.leaky_relu(_preactivate3_bn)
        tf.summary.histogram('activations', _layer3)
        _dropout3 = tf.nn.dropout(_layer3, rate=dropout_rate)

    print('layer1: {}'.format(_layer1))
    print('layer2: {}'.format(_layer2))
    print('layer2.5: {}'.format(_layer2_5))
    print('layer3: {}'.format(_layer3))

    # reshape to 1D tensor
    _layer_flat = tf.reshape(_dropout3, [-1, hidden3])

    # add last layer
    with tf.name_scope('final_layer'):
        with tf.name_scope('weights'):
            _w4 = tf.get_variable("final_layer_weight", shape=[hidden3, 1],
                                        initializer=tf.contrib.layers.xavier_initializer())
                                        # initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.add_to_collection('weight', _w4)
            variable_summaries(_w4)
        with tf.name_scope('biases'):
            _b4 = tf.get_variable("final_layer_bias", shape=[1],
                                initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('bias', _b4)
            variable_summaries(_b4)

        # apply final layer
        _pred_ka_values = tf.add(tf.matmul(_layer_flat, _w4), _b4, name='pred_ka')

    _cnn_weights = {
        'w1': _w1,
        'w2': _w2,
        'w3': _w3,
        'w4': _w4,
        'b4': _b4,
    }
    return _pred_ka_values, _cnn_weights


def pad_vals(vals, split_sizes, num_mirs, batch_size, fill_val):

    # get padding dimensions
    split_sizes_expand = tf.expand_dims(split_sizes, 1)
    paddings = tf.concat([tf.zeros(shape=tf.shape(split_sizes_expand), dtype=tf.int32),
                                      tf.reduce_max(split_sizes) - split_sizes_expand], axis=1)

    # split repression data and pad into batch_size_biochem x num_mirs x max_size matrix
    vals_split = tf.split(vals, tf.reshape(split_sizes, [num_mirs * batch_size]))

    # apply pads, fill with fill_val
    fill_val = tf.constant(fill_val, dtype=tf.float32)
    vals_split_padded = [tf.pad(vals_split[ix], paddings[ix:ix + 1, :], constant_values=fill_val) for ix in range(batch_size * num_mirs)]
    vals_reshaped = tf.reshape(tf.stack(vals_split_padded), [batch_size, num_mirs, -1])

    return vals_reshaped

# def pad_vals2(vals, split_sizes, num_mirs, batch_size, fill_val):

#     # get dense shape
#     dense_shape = tf.stack([tf.constant(batch_size, dtype=tf.int64), tf.constant(num_mirs, dtype=tf.int64), tf.cast(tf.reduce_max(split_sizes), dtype=tf.int64)])
    
#     # get dense indices
#     mask = tf.greater(tf.reshape(split_sizes, [-1, 1]), tf.reshape(tf.range(tf.reduce_max(split_sizes)), [1, -1]))
#     mask = tf.reshape(mask, [batch_size, num_mirs, -1])
#     dense_indices = tf.where(mask)

#     padded = tf.sparse.to_dense(tf.sparse.SparseTensor(dense_indices, vals, dense_shape), default_value=fill_val)
#     return padded, tf.cast(mask, tf.float32)

def get_pred_logfc_occupancy_only(_utr_ka_values, _freeAGO_all, _tpm_batch, _ts7_weights, _ts7_bias, _decay, batch_size, passenger, num_guides, name, loss_type):
    if passenger:
        num_mirs = num_guides * 2
    else:
        num_mirs = num_guides

    # merge with other features
    _weighted_features = tf.squeeze(tf.matmul(_tpm_batch['features'], _ts7_weights))
    _utr_ka_values_squeezed = tf.squeeze(_utr_ka_values)
    # _merged_features = tf.squeeze(_utr_ka_values) + _weighted_features # + _ts7_bias

    # pad values
    _weighted_features_padded = pad_vals(_weighted_features, _tpm_batch['nsites'], num_mirs, batch_size, fill_val=-100.0)
    _ka_vals_padded = pad_vals(_utr_ka_values_squeezed, _tpm_batch['nsites'], num_mirs, batch_size, fill_val=-100.0)

    _merged_features_mask = tf.cast(tf.greater(_ka_vals_padded, -50.0), tf.float32)
    _nbound_init = tf.sigmoid(_weighted_features_padded + tf.reshape(_freeAGO_all, [1, -1, 1]))
    _nbound = tf.sigmoid((_weighted_features_padded + _ka_vals_padded) + tf.reshape(_freeAGO_all, [1, -1, 1]))
    _nbound_net = _nbound - _nbound_init

    _masked_nbound = tf.multiply(_nbound_net, _merged_features_mask)
    # _masked_nbound = tf.sigmoid(_merged_features_padded + tf.reshape(_freeAGO_all, [1, -1, 1]))

    # calculate occupancy
    _occupancy = tf.exp(_decay) * tf.reduce_sum(_masked_nbound, axis=2)
    # _occupancy = tf.reduce_sum(_masked_nbound, axis=2)
    # Add guide and passenger strand occupancies, if applicable
    if passenger:
        _occupancy = tf.reduce_sum(tf.reshape(_occupancy, [-1, num_guides, 2]), axis=2)

    # get logfc
    _pred_logfc = -1 * tf.log1p(_occupancy)

    if loss_type == 'MEAN_CENTER':
        _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1, 1])
        _repression_y_normed = _tpm_batch['labels'] - tf.reshape(tf.reduce_mean(_tpm_batch['labels'], axis=1), [-1, 1])
    else:
        _pred_logfc_normed = _pred_logfc
        _repression_y_normed = _tpm_batch['labels']

    return _pred_logfc, _pred_logfc_normed, _repression_y_normed, (_masked_nbound)


# def get_pred_logfc_occupancy_only2(_utr_ka_values, _freeAGO_all, _tpm_batch, _ts7_weights, _ts7_bias, _decay, batch_size, passenger, num_guides, name, loss_type):
#     if passenger:
#         num_mirs = num_guides * 2
#     else:
#         num_mirs = num_guides

#     # merge with other features
#     _weighted_features = tf.squeeze(tf.matmul(_tpm_batch['features'], _ts7_weights))
#     _merged_features = tf.squeeze(_utr_ka_values) + _weighted_features # + _ts7_bias

#     # pad values
#     _weighted_features_padded, _ = pad_vals2(_weighted_features, _tpm_batch['nsites'], num_mirs, batch_size, fill_val=-100.0)
#     _merged_features_padded, _merged_features_mask = pad_vals2(_merged_features, _tpm_batch['nsites'], num_mirs, batch_size, fill_val=-100.0)

#     # _merged_features_mask = tf.cast(tf.greater(_ka_vals_padded, 0), tf.float32)
#     _nbound_init = tf.sigmoid(_weighted_features_padded + tf.reshape(_freeAGO_all, [1, -1, 1]))
#     _nbound = tf.sigmoid(_merged_features_padded + tf.reshape(_freeAGO_all, [1, -1, 1]))
#     _nbound_net = _nbound - _nbound_init

#     _masked_nbound = tf.multiply(_nbound_net, _merged_features_mask)

#     # calculate occupancy
#     _occupancy = tf.exp(_decay) * tf.reduce_sum(_masked_nbound, axis=2)

#     # Add guide and passenger strand occupancies, if applicable
#     if passenger:
#         _occupancy = tf.reduce_sum(tf.reshape(_occupancy, [-1, num_guides, 2]), axis=2)

#     # get logfc
#     _pred_logfc = -1 * tf.log1p(_occupancy)

#     if loss_type == 'MEAN_CENTER':
#         _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1, 1])
#         _repression_y_normed = _tpm_batch['labels'] - tf.reshape(tf.reduce_mean(_tpm_batch['labels'], axis=1), [-1, 1])
#     else:
#         _pred_logfc_normed = _pred_logfc
#         _repression_y_normed = _tpm_batch['labels']

#     return _pred_logfc, _pred_logfc_normed, _repression_y_normed, (_masked_nbound)


def get_pred_logfc_separate(_utr_ka_values, _freeAGO_all, _tpm_batch, _ts7_weights, _ts7_bias, _decay, batch_size, passenger, num_guides, name, loss_type):
    if passenger:
        num_mirs = num_guides * 2
    else:
        num_mirs = num_guides

    # merge with other features
    _merged_features = tf.squeeze(tf.matmul(_tpm_batch['features'], _ts7_weights)) + _ts7_bias

    # pad values
    _utr_ka_values_padded = pad_vals(tf.squeeze(_utr_ka_values), _tpm_batch['nsites'], num_mirs, batch_size)
    _merged_features_padded = tf.nn.relu(pad_vals(_merged_features, _tpm_batch['nsites'], num_mirs, batch_size))

    # calculate logfc
    _occupancy = tf.sigmoid(_utr_ka_values_padded + tf.reshape(_freeAGO_all, [1, -1, 1]))
    _pred_logfc = _decay * tf.squeeze(tf.reduce_sum(tf.multiply(_occupancy, _merged_features_padded), axis=2))

    if passenger:
        _pred_logfc = tf.reduce_sum(tf.reshape(_pred_logfc, [-1, 2]), axis=1)

    _pred_logfc = tf.reshape(_pred_logfc, [batch_size, num_guides])

    if loss_type == 'MEAN_CENTER':
        _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1, 1])
        _repression_y_normed = _tpm_batch['labels'] - tf.reshape(tf.reduce_mean(_tpm_batch['labels'], axis=1), [-1, 1])
    else:
        _pred_logfc_normed = _pred_logfc
        _repression_y_normed = _tpm_batch['labels']

    return _pred_logfc, _pred_logfc_normed, _repression_y_normed


def get_pred_logfc(_utr_ka_values, _freeAGO_all, _tpm_batch, _ts7_weights, batch_size, passenger, num_guides, name, loss_type):
    if passenger:
        # nsites_mir = tf.reduce_sum(tf.reshape(_tpm_batch['nsites'], [num_guides * batch_size, 2]), axis=1)
        num_mirs = num_guides * 2
    else:
        # nsites_mir = tf.reshape(_tpm_batch['nsites'], [num_guides * batch_size])
        num_mirs = num_guides

    # merge with other features
    _merged_features = tf.concat([_utr_ka_values, _tpm_batch['features']], axis=1)
    _merged_features = (tf.squeeze(tf.matmul(_merged_features, _ts7_weights)))

    # pad ka values
    _merged_features_reshaped = pad_vals(_merged_features, _tpm_batch['nsites'], num_mirs, batch_size)

    # calculate logfc
    _pred_logfc = -1 * tf.squeeze(tf.reduce_sum(tf.sigmoid(_merged_features_reshaped + tf.reshape(_freeAGO_all, [1, -1, 1])), axis=2))

    if passenger:
        _pred_logfc = tf.reduce_sum(tf.reshape(_pred_logfc, [-1, 2]), axis=1)

    _pred_logfc = tf.reshape(_pred_logfc, [batch_size, num_guides])

    if loss_type == 'MEAN_CENTER':
        _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1, 1])
        _repression_y_normed = _tpm_batch['labels'] - tf.reshape(tf.reduce_mean(_tpm_batch['labels'], axis=1), [-1, 1])
    else:
        _pred_logfc_normed = _pred_logfc
        _repression_y_normed = _tpm_batch['labels']

    return _pred_logfc, _pred_logfc_normed, _repression_y_normed

