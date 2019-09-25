from optparse import OptionParser
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tensorflow as tf

import model
import parse_data_utils
import utils

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


def get_pred_logfc_tsfeats(_utr_ka_values, _freeAGO_all, _tpm_batch, _all_site_weights, _canon_site_weights,
                                _decay, batch_size, passenger, num_guides, name, loss_type):
    if passenger:
        num_mirs = num_guides * 2
    else:
        num_mirs = num_guides

    # merge with other features
    all_ixs = [0,1,2,4]
    canon_ixs = [3,5]
    _weighted_features_all = tf.squeeze(tf.matmul(_tpm_batch['features'][:,all_ixs], _all_site_weights))
    _weighted_features_canon = tf.squeeze(tf.matmul(_tpm_batch['features'][:,canon_ixs], _canon_site_weights))
    _utr_ka_values_squeezed = tf.squeeze(_utr_ka_values)
    # _merged_features = tf.squeeze(_utr_ka_values) + _weighted_features # + _ts7_bias

    # pad values
    _weighted_features_all_padded = pad_vals(_weighted_features_all, _tpm_batch['nsites'], num_mirs, batch_size, fill_val=-100.0)
    _weighted_features_canon_padded = pad_vals(_weighted_features_canon, _tpm_batch['nsites'], num_mirs, batch_size, fill_val=-100.0)
    _ka_vals_padded = pad_vals(_utr_ka_values_squeezed, _tpm_batch['nsites'], num_mirs, batch_size, fill_val=-100.0)

    _merged_features_mask = tf.cast(tf.greater(_ka_vals_padded, -50.0), tf.float32)
    _nbound_init = tf.sigmoid(_weighted_features_all_padded + tf.reshape(_freeAGO_all, [1, -1, 1]))
    _nbound = tf.sigmoid((_weighted_features_all_padded + _weighted_features_canon_padded + _ka_vals_padded) + tf.reshape(_freeAGO_all, [1, -1, 1]))
    _nbound_net = _nbound - _nbound_init

    _masked_nbound = tf.multiply(_nbound_net, _merged_features_mask)

    # calculate occupancy
    _occupancy = tf.exp(_decay) * tf.reduce_sum(_masked_nbound, axis=2)

    if passenger:
        _occupancy = tf.reduce_sum(tf.reshape(_occupancy, [-1, num_guides, 2]), axis=2)

    # get logfc
    _pred_logfc = -1 * tf.log1p(_occupancy, name=name)

    if loss_type == 'MEAN_CENTER':
        _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1, 1])
        _repression_y_normed = _tpm_batch['labels'] - tf.reshape(tf.reduce_mean(_tpm_batch['labels'], axis=1), [-1, 1])
    else:
        _pred_logfc_normed = _pred_logfc
        _repression_y_normed = _tpm_batch['labels']

    return _pred_logfc, _pred_logfc_normed, _repression_y_normed, (_masked_nbound)


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_tfrecords", dest="TPM_TFRECORDS", help="tpm data in tfrecord format")
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--freeagos", dest="FREEAGOS", help="tsv with miRNAs and freeAGO concentrations", default=None)
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--num_feats", dest="NUM_FEATS", type=int)
    parser.add_option("--use_feats", dest="USE_FEATS", type=int)
    parser.add_option("--loss_type", dest="LOSS_TYPE", help="which loss strategy")
    parser.add_option("--logdir", dest="LOGDIR", help="directory for writing outputs")
    parser.add_option("--load_model", dest="LOAD_MODEL", help="directory to load model from")
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')

    (options, args) = parser.parse_args()

    tf.reset_default_graph()

    if options.LOGDIR is not None:
        # make log directory if it doesn't exist
        if (not os.path.isdir(options.LOGDIR)):
            os.makedirs(options.LOGDIR)

    # SEQLEN must be 12
    SEQLEN = 12

    ### READ miRNA DATA ###
    MIRNA_DATA = pd.read_csv(options.MIR_SEQS, sep='\t', index_col='mir')
    MIRNA_DATA_WITH_RBNS = MIRNA_DATA[MIRNA_DATA['has_rbns']]
    MIRNA_DATA_USE_TPMS = MIRNA_DATA[MIRNA_DATA['use_tpms']]

    ALL_GUIDES = sorted(list(MIRNA_DATA_USE_TPMS.index))

    # TPM data reader
    tpm_val_dataset = tf.data.TFRecordDataset(options.TPM_TFRECORDS)

    def _parse_fn(x):
        return parse_data_utils._parse_repression_function(x, ALL_GUIDES, options.MIRLEN, SEQLEN, options.NUM_FEATS, options.USE_FEATS, options.PASSENGER)

    # preprocess data
    tpm_val_dataset = tpm_val_dataset.map(_parse_fn, num_parallel_calls=16)

    # make feedable iterators
    tpm_val_iterator = tpm_val_dataset.make_initializable_iterator()

    # create handle for switching between training and validation
    tpm_handle = tf.placeholder(tf.string, shape=[])
    tpm_iterator = tf.data.Iterator.from_string_handle(tpm_handle, tpm_val_dataset.output_types)
    next_tpm_batch = parse_data_utils._build_tpm_batch(tpm_iterator, 1)

    FREEAGOS = pd.read_csv(options.FREEAGOS, sep='\t', index_col=0)
    if options.PASSENGER:
        FREEAGOS = FREEAGOS.loc[ALL_GUIDES][['guide','pass']].values.flatten()
    else:
        FREEAGOS = FREEAGOS.loc[ALL_GUIDES][['guide']].values.flatten()

    print(FREEAGOS.shape)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(options.LOAD_MODEL + '.meta')
        saver.restore(sess, options.LOAD_MODEL)

        # for op in sess.graph.get_operations():
        #     if 'ts7_weights' in op.name:
        #         print(op.name)

        _dropout_rate = tf.get_default_graph().get_tensor_by_name('dropout_rate:0')
        _phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        _combined_x = tf.get_default_graph().get_tensor_by_name('combined_x:0')
        _utr_ka_values = tf.get_default_graph().get_tensor_by_name('final_layer/pred_ka:0')
        _decay = tf.get_default_graph().get_tensor_by_name('decay:0')

        # in_orf, mindist/2000, local_au, sa_score
        # threep, pct

        all_site_weights_init = np.array([-2.22, -0.2, 0.1, 0.1]).reshape([4, 1])
        canon_site_weights_init = np.array([0.1, 0.1]).reshape([2, 1])
        decay_init = 0.0

        # create ts7 weight variable
        with tf.name_scope('ts7_layer'):
            with tf.name_scope('weights'):
                _all_site_weights = tf.get_variable("all_site_weights", shape=[4, 1],
                                            initializer=tf.constant_initializer(all_site_weights_init))

                _canon_site_weights = tf.get_variable("all_site_weights", shape=[2, 1],
                                            initializer=tf.constant_initializer(canon_site_weights_init))

                _decay = tf.get_variable('decay', initializer=decay_init)

        _freeAGO_all_val = tf.constant(FREEAGOS, dtype=tf.float32)

        _utr_ka_values, _freeAGO_all, _tpm_batch, _all_site_weights, _canon_site_weights,
                                _decay, batch_size, passenger, num_guides, name, loss_type

        _pred_logfc_val, _pred_logfc_val_normed, _repression_y_val_normed, _ = model.get_pred_logfc_occupancy_only(
            _utr_ka_values,
            _freeAGO_all_val,
            next_tpm_batch,
            _all_site_weights,
            _canon_site_weights,
            _decay,
            options.BATCH_SIZE,
            options.PASSENGER,
            len(ALL_GUIDES),
            'pred_logfc_val',
            options.LOSS_TYPE
        )


        sess.run(tpm_val_iterator.initializer)
        tpm_val_handle = sess.run(tpm_val_iterator.string_handle())

        transcripts, pred_vals, real_vals, real_nsites = [], [], [], []
        while True:
            try:
                temp_tpm_batch = sess.run(next_tpm_batch, feed_dict={tpm_handle: tpm_val_handle})
                transcripts.append(temp_tpm_batch['transcripts'])
                real_vals.append(temp_tpm_batch['labels'])
                ka_vals = sess.run(_utr_ka_values, feed_dict={_phase_train: False, _dropout_rate: 0.0, _combined_x: temp_tpm_batch['images']})
                pred_vals.append(sess.run(_pred_logfc_val,
                    feed_dict={
                        _utr_ka_values: ka_vals,
                        next_tpm_batch['nsites']: temp_tpm_batch['nsites'],
                        next_tpm_batch['features']: temp_tpm_batch['features'],
                        next_tpm_batch['labels']: temp_tpm_batch['labels'],
                    }))
            except tf.errors.OutOfRangeError:
                break

        transcripts = np.concatenate(transcripts)
        pred_vals = np.concatenate(pred_vals)
        real_vals = np.concatenate(real_vals)

        pred_vals_normed = pred_vals - np.mean(pred_vals, axis=1).reshape([-1,1])
        real_vals_normed = real_vals - np.mean(real_vals, axis=1).reshape([-1,1])

        pred_df = pd.DataFrame({
            'transcript': np.repeat(transcripts.flatten(), len(ALL_GUIDES)),
            'mir': list(ALL_GUIDES) * len(transcripts),
            'pred': pred_vals.flatten(),
            'label': real_vals.flatten(),
            'pred_normed': pred_vals_normed.flatten(),
            'label_normed': real_vals_normed.flatten()
        })

        pred_df.to_csv(os.path.join(options.LOGDIR, 'pred_df_all.txt'), sep='\t', index=False)
