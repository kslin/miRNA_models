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
    parser.add_option("--test_mirna", dest="TEST_MIRNA", help="testing miRNA")
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

    ### READ miRNA DATA ###
    MIRNA_DATA = pd.read_csv(options.MIR_SEQS, sep='\t')
    ALL_GUIDES = list(MIRNA_DATA['mir'].values)

    # split miRNAs into training and testing
    if options.TEST_MIRNA == 'none':
        TRAIN_GUIDES = ALL_GUIDES
        TEST_GUIDES = ['mir139']
    else:
        if options.TEST_MIRNA not in ALL_GUIDES:
            raise ValueError('Test miRNA not in mirseqs file.')
        TRAIN_GUIDES = [m for m in ALL_GUIDES if m != options.TEST_MIRNA]
        TEST_GUIDES = [options.TEST_MIRNA]

    if options.PASSENGER:
        TRAIN_MIRS = np.array(list(zip(TRAIN_GUIDES, [x+'*' for x in TRAIN_GUIDES]))).flatten().tolist()
        ALL_MIRS = np.array(list(zip(ALL_GUIDES, [x+'*' for x in ALL_GUIDES]))).flatten().tolist()
    else:
        TRAIN_MIRS = TRAIN_GUIDES
        ALL_MIRS = ALL_GUIDES

    # TPM data reader
    raw_tpm_dataset = tf.data.TFRecordDataset(options.TPM_TFRECORDS)
    _parse_fn = lambda x: parse_data._parse_repression_function(x, TRAIN_MIRS, ALL_MIRS, options.MIRLEN, SEQLEN, options.NUM_FEATS)
    parsed_tpm_dataset = raw_tpm_dataset.map(_parse_fn)
    iterator_tpm = parsed_tpm_dataset.make_initializable_iterator()
    next_tpm_batch = [iterator_tpm.get_next() for _ in range(options.REPRESSION_BATCH_SIZE)]


    # KD data reader
    raw_kd_dataset = tf.data.TFRecordDataset(options.KD_TFRECORDS)
    parsed_kd_dataset = raw_kd_dataset.map(parse_data._parse_log_kd_function)
    parsed_kd_dataset = parsed_kd_dataset.batch(options.KD_BATCH_SIZE)
    iterator_kd = parsed_kd_dataset.make_initializable_iterator()
    next_kd_batch_x, next_kd_batch_y = iterator_kd.get_next()

    
