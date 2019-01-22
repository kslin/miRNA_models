from optparse import OptionParser
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import config
import data_objects
import helpers
import tf_helpers

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_tfrecords", dest="TPM_TFRECORDS", help="tpm data in tfrecord format")
    parser.add_option("--kd_tfrecords", dest="KD_TFRECORDS", help="kd data in tfrecord format")
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--num_feats", dest="NUM_FEATS", type=int)
    parser.add_option("--repression_batch_size", dest="REPRESSION_BATCH_SIZE", type=int)
    parser.add_option("--kd_batch_size", dest="KD_BATCH_SIZE", type=int)
    parser.add_option("--test_mir", dest="TEST_MIR", help="testing miRNA")
    parser.add_option("--baseline", dest="BASELINE_METHOD", help="which baseline to use")
    parser.add_option("--loss_type", dest="LOSS_TYPE", help="which loss strategy")
    parser.add_option("--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("--load_model", dest="LOAD_MODEL", help="if supplied, load latest model from this directory", default=None)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')

    (options, args) = parser.parse_args()

    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    # SEQLEN must be 12
    SEQLEN = 12
    REPRESSION_WEIGHT = 50.0 * KD_BATCH_SIZE / REPRESSION_BATCH_SIZE

    ### READ miRNA DATA ###
    MIRNA_DATA = pd.read_csv(options.MIR_SEQS, sep='\t')
    ALL_GUIDES = list(MIRNA_DATA['mir'].values)

    # split miRNAs into training and testing
    if options.TEST_MIRNA == 'none':
        TRAIN_GUIDES = ALL_GUIDES
    else:
        if options.TEST_MIRNA not in ALL_GUIDES:
            raise ValueError('Test miRNA not in mirseqs file.')
        TRAIN_GUIDES = [m for m in ALL_GUIDES if m != options.TEST_MIRNA]

    NUM_TRAIN = len(TRAIN_GUIDES)

    if options.PASSENGER:
        TRAIN_MIRS = np.array(list(zip(TRAIN_GUIDES, [x+'*' for x in TRAIN_GUIDES]))).flatten().tolist()
        ALL_MIRS = np.array(list(zip(ALL_GUIDES, [x+'*' for x in ALL_GUIDES]))).flatten().tolist()
    else:
        TRAIN_MIRS = TRAIN_GUIDES
        ALL_MIRS = ALL_GUIDES

    # TPM data reader
    tpm_dataset = tf.data.TFRecordDataset(options.TPM_TFRECORDS)
    tpm_dataset = tpm_dataset.shuffle(buffer_size=1000)
    _parse_fn = lambda x: parse_data._parse_repression_function(x, TRAIN_MIRS, ALL_MIRS, options.MIRLEN, SEQLEN, options.NUM_FEATS)
    tpm_dataset = tpm_dataset.map(_parse_fn)
    tpm_iterator = tpm_dataset.make_initializable_iterator()

    # build tpm batch
    next_tpm_batch = parse_data._build_tpm_batch(tpm_iterator, options.REPRESSION_BATCH_SIZE)

    # KD data reader
    kd_dataset = tf.data.TFRecordDataset(options.KD_TFRECORDS)
    kd_dataset = kd_dataset.shuffle(buffer_size=1000)
    kd_dataset = kd_dataset.map(parse_data._parse_log_kd_function)

    # split into train and test
    kd_train_dataset = kd_dataset.filter(lambda x, y, z: tf.math.logical_not(tf.equal(x, options.TEST_MIR.encode('utf-8'))))
    kd_test_dataset = kd_dataset.filter(lambda x, y, z: tf.equal(x, options.TEST_MIR.encode('utf-8')))

    # batch train datasets
    kd_train_dataset = kd_train_dataset.batch(options.KD_BATCH_SIZE)
    kd_train_iterator = kd_train_dataset.make_initializable_iterator()
    next_kd_train_batch_mirs, next_kd_train_batch_images, next_kd_train_batch_labels = kd_train_iterator.get_next()

    # batch test datasets
    kd_test_dataset = kd_test_dataset.batch(1000)
    kd_test_iterator = kd_test_dataset.make_initializable_iterator()
    next_kd_test_batch_mirs, next_kd_test_batch_images, next_kd_test_batch_labels = kd_test_iterator.get_next()

    # create placeholders for input data
    _keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    _phase_train = tf.placeholder(tf.bool, name='phase_train')

    # build KA predictor
    _combined_x = tf.concat([next_tpm_batch['images'], next_kd_test_batch_images], axis=0)
    _combined_x_4D = tf.expand_dims((_combined_x * 4.0) - 0.25, axis=3)  # reshape, zero-center input
    _pred_ka_values, _cnn_weights = tf_helpers.seq2ka_predictor(_combined_x_4D, _keep_prob, _phase_train)  # pred ka

    # split data into biochem and repression and get biochem loss
    if options.KD_BATCH_SIZE == 0:
        _pred_biochem = tf.constant(np.array([[0]]))
        _biochem_loss = tf.constant(0.0)
        _utr_ka_values = tf.reshape(_pred_ka_values, [-1])
    else:
        _pred_biochem = _pred_ka_values[-1 * options.KD_BATCH_SIZE:, :]
        _biochem_loss = (tf.nn.l2_loss(tf.subtract(_pred_biochem, _biochem_y)))
        _utr_ka_values = tf.reshape(_pred_ka_values[:-1 * options.KD_BATCH_SIZE, :], [-1])

    # reshape repression ka values
    _utr_max_size = tf.reduce_max(next_tpm_batch['nsites'])
    _utr_ka_values_reshaped = tf_helpers.pad_kd_from_genes(
        _utr_ka_values, next_tpm_batch['nsites'], _utr_max_size, NUM_TRAIN, options.REPRESSION_BATCH_SIZE, options.PASSENGER
    )

    print('utr_ka_reshaped: {}'.format(_utr_ka_values_reshaped))

    # get repression prediction
    init_params = [
        -4.0,  # FREEAGO_INIT,
        0.0,   # GUIDE_OFFSET_INIT,
        -1.0,  # PASS_OFFSET_INIT,
        -0.5,  # DECAY_INIT,
        -8.5,  # UTR_COEF_INIT
    ]

    _results = tf_helpers.ka2repression_predictor(
        'train',
        _utr_ka_values_reshaped,
        _utr_len,
        NUM_TRAIN,
        config.BATCH_SIZE_REPRESSION,
        init_params
    )

    _pred_logfc = _results['pred_logfc_net']




    
