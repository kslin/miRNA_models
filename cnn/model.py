import numpy as np
import pandas as pd
import tensorflow as tf

import parse_data_utils

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

def seq2ka_predictor(input_data, keep_prob, phase_train, hidden1, hidden2, hidden3, mirlen, seqlen):
    # add layer 1
    with tf.name_scope('layer1'):
        _w1, _b1 = get_conv_params(4, 4, 1, hidden1, 'layer1')
        _preactivate1 = tf.nn.conv2d(input_data, _w1, strides=[1, 4, 4, 1], padding='VALID') + _b1
        _preactivate1_bn = tf.layers.batch_normalization(_preactivate1, training=phase_train)
        _layer1 = tf.nn.leaky_relu(_preactivate1_bn)

    # add layer 2
    with tf.name_scope('layer2'):
        _w2, _b2 = get_conv_params(2, 2, hidden1, hidden2, 'layer2')
        _preactivate2 = tf.nn.conv2d(_layer1, _w2, strides=[1, 1, 1, 1], padding='VALID') + _b2
        _preactivate2_bn = tf.layers.batch_normalization(_preactivate2, training=phase_train)
        _layer2 = tf.nn.leaky_relu(_preactivate2_bn)
        _dropout2 = tf.nn.dropout(_layer2, rate=1.0 - keep_prob)

    # add layer 3
    with tf.name_scope('layer3'):
        _w3, _b3 = get_conv_params(mirlen - 1, seqlen - 1, hidden2, hidden3, 'layer3')
        _preactivate3 = tf.nn.conv2d(_dropout2, _w3, strides=[1, 1, 1, 1], padding='VALID') + _b3
        _preactivate3_bn = tf.layers.batch_normalization(_preactivate3, training=phase_train)
        _layer3 = tf.nn.leaky_relu(_preactivate3_bn)
        _dropout3 = tf.nn.dropout(_layer3, rate=1.0 - keep_prob)

    print('layer1: {}'.format(_layer1))
    print('layer2: {}'.format(_layer2))
    print('layer3: {}'.format(_layer3))

    # reshape to 1D tensor
    _layer_flat = tf.reshape(_dropout3, [-1, hidden3])

    # add last layer
    with tf.name_scope('final_layer'):
        with tf.name_scope('weights'):
            _w4 = tf.get_variable("final_layer_weight", shape=[hidden3, 1],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.add_to_collection('weight', _w4)
        with tf.name_scope('biases'):
            _b4 = tf.get_variable("final_layer_bias", shape=[1],
                                initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('bias', _b4)

        # apply final layer
        _pred_ka_values = tf.add(tf.matmul(_layer_flat, _w4), _b4, name='pred_ka')

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


def get_pred_logfc_occupancy_only(_utr_ka_values, _freeAGO_all, _tpm_batch, _ts7_weights, _ts7_bias, _decay, batch_size, passenger, num_guides, name, loss_type):
    if passenger:
        num_mirs = num_guides * 2
    else:
        num_mirs = num_guides

    # merge with other features
    _merged_features = tf.squeeze(_utr_ka_values + tf.matmul(_tpm_batch['features'], _ts7_weights))# + _ts7_bias

    # pad values
    _merged_features_padded = pad_vals(_merged_features, _tpm_batch['nsites'], num_mirs, batch_size, fill_val=-100.0)
    _merged_features_mask = tf.cast(tf.greater(_merged_features_padded, 0), tf.float32)

    _masked_nbound = tf.multiply(tf.sigmoid(_merged_features_padded + tf.reshape(_freeAGO_all, [1, -1, 1])), _merged_features_mask)

    # calculate occupancy
    _occupancy = tf.exp(_decay) * tf.reduce_sum(_masked_nbound, axis=2)

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

    return _pred_logfc, _pred_logfc_normed, _repression_y_normed, (_merged_features_padded, _masked_nbound, _occupancy, _pred_logfc)


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


def get_pred_logfc_old(_utr_ka_values, _freeAGO_all, _tpm_batch, _ts7_weights, batch_size, passenger, num_guides, name, loss_type):
    if passenger:
        nsites_mir = tf.reduce_sum(tf.reshape(_tpm_batch['nsites'], [num_guides * batch_size, 2]), axis=1)
        num_mirs = num_guides * 2
    else:
        nsites_mir = tf.reshape(_tpm_batch['nsites'], [num_guides * batch_size])
        num_mirs = num_guides

    _freeAGO_tiled = tf.tile(_freeAGO_all, tf.constant([batch_size]))
    _freeAGO_tiled = tf.concat([tf.tile(_freeAGO_tiled[ix: ix + 1], _tpm_batch['nsites'][ix: ix + 1]) for ix in range(num_mirs * batch_size)], axis=0)

    _merged_features = tf.concat([_utr_ka_values, _tpm_batch['features']], axis=1)
    _merged_features = (tf.matmul(_merged_features, _ts7_weights))

    _pred_logfc_ind_sites = -1 * tf.squeeze(tf.sigmoid(_merged_features + tf.expand_dims(_freeAGO_tiled, axis=1)))
    # _all_feats = tf.concat([_nbound, _tpm_batch['features']], axis=1)
    # _pred_logfc_ind_sites = -1 * (tf.squeeze(tf.matmul(_all_feats, _ts7_weights)))
    _pred_logfc_splits = tf.split(_pred_logfc_ind_sites, nsites_mir)
    _pred_logfc = tf.reshape(tf.stack([tf.reduce_sum(x) for x in _pred_logfc_splits]), [batch_size, num_guides], name=name)

    if loss_type == 'MEAN_CENTER':
        _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1, 1])
        _repression_y_normed = _tpm_batch['labels'] - tf.reshape(tf.reduce_mean(_tpm_batch['labels'], axis=1), [-1, 1])
    else:
        _pred_logfc_normed = _pred_logfc
        _repression_y_normed = _tpm_batch['labels']

    return _pred_logfc, _pred_logfc_normed, _repression_y_normed
    # return _pred_logfc


# class MirModel:
#     def __init__(self, mirlen, num_feats, seqlen, train_mirs, all_mirs, rbns_train_mirs, tpm_batch_size, kd_batch_size):
#         self.mirlen = mirlen
#         self.seqlen = seqlen
#         self.num_feats = num_feats
#         self.train_mirs = train_mirs.flatten()
#         self.all_mirs = all_mirs.flatten()
#         self.num_train_mirs = len(train_mirs.flatten())
#         self.num_train_experiments = train_mirs.shape[0]
#         self.num_experiments = all_mirs.shape[0]
#         self.rbns_train_mirs = rbns_train_mirs
#         self.tpm_batch_size = tpm_batch_size
#         self.kd_batch_size = kd_batch_size

#         self.vars = {}
#         self.tensors = {}

#         self.vars['freeAGO_mean'] = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(-4.0))
#         self.vars['freeAGO_offset'] = tf.get_variable('freeAGO_offset', shape=[self.num_train_mirs], initializer=tf.constant_initializer(0.0))
#         self.vars['freeAGO_all'] = self.vars['freeAGO_mean'] + self.vars['freeAGO_offset']

#         # create placeholders for input data
#         self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#         self.phase_train = tf.placeholder(tf.bool, name='phase_train')

#         # create ts7 weight variable
#         with tf.name_scope('ts7_layer'):
#             with tf.name_scope('weights'):
#                 ts7_weights = tf.get_variable("ts7_weights", shape=[self.num_feats + 1, 1],
#                                             initializer=tf.truncated_normal_initializer(stddev=0.1))
#                 tf.add_to_collection('weight', ts7_weights)

#         self.vars['ts7_weights'] = ts7_weights


#     def add_tpm_data(self, tpm_dataset):

#         tpm_train_dataset = tpm_dataset.shuffle(buffer_size=1000)

#         def _parse_fn_train(x):
#             return parse_data_utils._parse_repression_function(x, self.train_mirs, self.all_mirs, self.mirlen, self.seqlen, self.num_feats)

#         def _parse_fn_val(x):
#             return parse_data_utils._parse_repression_function(x, self.all_mirs, self.all_mirs, self.mirlen, self.seqlen, self.num_feats)

#         # preprocess data
#         tpm_train_dataset = tpm_train_dataset.map(_parse_fn_train)
#         tpm_val_dataset = tpm_dataset.map(_parse_fn_val)

#         # make feedable iterators
#         self.tpm_train_iterator = tpm_train_dataset.make_initializable_iterator()
#         self.tpm_val_iterator = tpm_val_dataset.make_initializable_iterator()

#         # create handle for switching between training and validation
#         self.tpm_handle = tf.placeholder(tf.string, shape=[])
#         tpm_iterator = tf.data.Iterator.from_string_handle(tpm_handle, tpm_train_dataset.output_types)
#         self.next_tpm_batch = parse_data_utils._build_tpm_batch(tpm_iterator, self.tpm_batch_size)


#     def add_kd_data(self, kd_train_dataset, kd_val_dataset):

#         # shuffle, batch, and map datasets
#         kd_train_dataset = kd_train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=None)
#         kd_train_dataset = kd_train_dataset.map(parse_data_utils._parse_log_kd_function)
#         kd_train_dataset = kd_train_dataset.batch(self.kd_batch_size, drop_remainder=True)

#         kd_val_dataset = kd_val_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=None)
#         kd_val_dataset = kd_val_dataset.map(parse_data_utils._parse_log_kd_function)
#         kd_val_dataset = kd_val_dataset.batch(1000, drop_remainder=True)

#         # make feedable iterators
#         self.kd_train_iterator = kd_train_dataset.make_initializable_iterator()
#         self.kd_val_iterator = kd_val_dataset.make_initializable_iterator()

#         # create handle for switching between training and validation
#         self.kd_handle = tf.placeholder(tf.string, shape=[])
#         kd_iterator = tf.data.Iterator.from_string_handle(kd_handle, kd_train_dataset.output_types)
#         next_kd_batch_mirs, next_kd_batch_images, next_kd_batch_labels = kd_iterator.get_next()

#         self.next_kd_batch = {
#             'mirs': next_kd_batch_mirs,
#             'images': next_kd_batch_images,
#             'raw_labels': next_kd_batch_labels,
#             'labels': tf.nn.relu(-1 * next_kd_batch_labels)
#         }


#     def _predict_ka(self, input_data):
#         # add layer 1
#         with tf.name_scope('layer1'):
#             _w1, _b1 = get_conv_params(4, 4, 1, self.hidden1, 'layer1')
#             _preactivate1 = tf.nn.conv2d(input_data, _w1, strides=[1, 4, 4, 1], padding='VALID') + _b1
#             _preactivate1_bn = tf.layers.batch_normalization(_preactivate1, training=self.phase_train)
#             _layer1 = tf.nn.leaky_relu(_preactivate1_bn)

#         # add layer 2
#         with tf.name_scope('layer2'):
#             _w2, _b2 = get_conv_params(2, 2, self.hidden1, self.hidden2, 'layer2')
#             _preactivate2 = tf.nn.conv2d(_layer1, _w2, strides=[1, 1, 1, 1], padding='VALID') + _b2
#             _preactivate2_bn = tf.layers.batch_normalization(_preactivate2, training=self.phase_train)
#             _layer2 = tf.nn.leaky_relu(_preactivate2_bn)
#             _dropout2 = tf.nn.dropout(_layer2, self.keep_prob)

#         # add layer 3
#         with tf.name_scope('layer3'):
#             _w3, _b3 = get_conv_params(self.mirlen - 1, self.seqlen - 1, self.hidden2, self.hidden3, 'layer3')
#             _preactivate3 = tf.nn.conv2d(_dropout2, _w3, strides=[1, 1, 1, 1], padding='VALID') + _b3
#             _preactivate3_bn = tf.layers.batch_normalization(_preactivate3, training=self.phase_train)
#             _layer3 = tf.nn.leaky_relu(_preactivate3_bn)
#             _dropout3 = tf.nn.dropout(_layer3, self.keep_prob)

#         print('layer1: {}'.format(_layer1))
#         print('layer2: {}'.format(_layer2))
#         print('layer3: {}'.format(_layer3))

#         # reshape to 1D tensor
#         _layer_flat = tf.reshape(_dropout3, [-1, self.hidden3])

#         # add last layer
#         with tf.name_scope('final_layer'):
#             with tf.name_scope('weights'):
#                 _w4 = tf.get_variable("final_layer_weight", shape=[self.hidden3, 1],
#                                             initializer=tf.truncated_normal_initializer(stddev=0.1))
#                 tf.add_to_collection('weight', _w4)
#             with tf.name_scope('biases'):
#                 _b4 = tf.get_variable("final_layer_bias", shape=[1],
#                                     initializer=tf.constant_initializer(0.0))
#                 tf.add_to_collection('bias', _b4)

#             # apply final layer
#             self.pred_ka_values = tf.nn.relu(tf.add(tf.matmul(_layer_flat, _w4), _b4), name='pred_ka')

#         self.vars['cnn_weights'] = {
#             'w1': _w1,
#             'b1': _b1,
#             'w2': _w2,
#             'b2': _b2,
#             'w3': _w3,
#             'b3': _b3,
#             'w4': _w4,
#             'b4': _b4,
#         }


    # def _predict_logfc(self):








