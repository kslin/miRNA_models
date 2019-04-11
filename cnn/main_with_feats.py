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


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_tfrecords", dest="TPM_TFRECORDS", help="tpm data in tfrecord format")
    parser.add_option("--kd_tfrecords", dest="KD_TFRECORDS", help="kd data in tfrecord format")
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--hidden1", dest="HIDDEN1", type=int)
    parser.add_option("--hidden2", dest="HIDDEN2", type=int)
    parser.add_option("--hidden3", dest="HIDDEN3", type=int)
    parser.add_option("--num_feats", dest="NUM_FEATS", type=int)
    parser.add_option("--use_feats", dest="USE_FEATS", type=int)
    parser.add_option("--repression_batch_size", dest="REPRESSION_BATCH_SIZE", type=int)
    parser.add_option("--kd_batch_size", dest="KD_BATCH_SIZE", type=int)
    parser.add_option("--num_epochs", dest="NUM_EPOCHS", type=int)
    parser.add_option("--val_mir", dest="VAL_MIR", help="testing miRNA")
    parser.add_option("--batch", dest="BATCH", help="which batch to withold from training", type=int)
    parser.add_option("--loss_type", dest="LOSS_TYPE", help="which loss strategy")
    parser.add_option("--lambda", dest="LAMBDA", help="regularizer weight", type=float)
    parser.add_option("--lr", dest="LEARNING_RATE", help="starting learning rate", type=float)
    parser.add_option("--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("--load_model", dest="LOAD_MODEL", help="if supplied, load latest model from this directory", default=None)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--dry_run", dest="DRY_RUN", help="if true, do dry run", default=False, action='store_true')
    parser.add_option("--pretrain", dest="PRETRAIN", help="if true, do pretraining step", default=False, action='store_true')

    (options, args) = parser.parse_args()

    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    tf.reset_default_graph()

    if (not os.path.isdir(options.LOGDIR)):
        os.makedirs(options.LOGDIR)


    test_case = np.array([[
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]])

    # SEQLEN must be 12
    SEQLEN = 12

    ### READ miRNA DATA ###
    MIRNA_DATA = pd.read_csv(options.MIR_SEQS, sep='\t', index_col='mir')
    MIRNA_DATA_WITH_RBNS = MIRNA_DATA[MIRNA_DATA['has_rbns']]
    MIRNA_DATA_USE_TPMS = MIRNA_DATA[MIRNA_DATA['use_tpms']]

    ALL_GUIDES = sorted(list(MIRNA_DATA_USE_TPMS.index))
    
    # # uncomment to overfit 
    # ALL_GUIDES = [x for x in ALL_GUIDES if x != options.VAL_MIR] + [options.VAL_MIR]
    # TRAIN_GUIDES = ALL_GUIDES
    # VAL_GUIDES = []

    # split miRNAs into training and testing
    if options.VAL_MIR == 'none':
        TRAIN_GUIDES = ALL_GUIDES
        VAL_GUIDES = []
    else:
        if options.VAL_MIR not in ALL_GUIDES:
            raise ValueError('Test miRNA not in mirseqs file.')
        TRAIN_GUIDES = [m for m in ALL_GUIDES if m != options.VAL_MIR]
        VAL_GUIDES = [options.VAL_MIR]

    if options.VAL_MIR in list(MIRNA_DATA_WITH_RBNS.index):
        TRAIN_MIRS_KDS = [x for x in list(MIRNA_DATA_WITH_RBNS.index) if x != options.VAL_MIR]
        VAL_MIRS_KDS = [options.VAL_MIR]
    elif options.VAL_MIR == 'none':
        TRAIN_MIRS_KDS = list(MIRNA_DATA_WITH_RBNS.index)
        VAL_MIRS_KDS = ['mir124']
    else:
        TRAIN_MIRS_KDS = list(MIRNA_DATA_WITH_RBNS.index)
        VAL_MIRS_KDS = [options.VAL_MIR]

    NUM_TRAIN_GUIDES = len(TRAIN_GUIDES)

    STYPE_COLOR_DICT = {
        b'8mer': 'red',
        b'7mer-m8': 'orange',
        b'7mer-a1': 'yellow',
        b'6mer': 'green',
        b'6mer-m8': 'blue',
        b'6mer-a1': 'purple',
        b'no site': 'grey'
    }

    print("Repression datasets for training: {}".format(TRAIN_GUIDES))
    print("Repression datasets for validating: {}".format(VAL_GUIDES))
    print("RBNS miRNAs for training: {}".format(TRAIN_MIRS_KDS))
    print("RBNS miRNAs for validating: {}".format(VAL_MIRS_KDS))

    # TPM data reader
    tpm_dataset = tf.data.TFRecordDataset(options.TPM_TFRECORDS)
    tpm_dataset = tpm_dataset.shuffle(buffer_size=1000)

    def _parse_fn_train(x):
        return parse_data_utils._parse_repression_function(x, TRAIN_GUIDES, options.MIRLEN, SEQLEN, options.NUM_FEATS, options.USE_FEATS, options.PASSENGER)

    def _parse_fn_val(x):
        return parse_data_utils._parse_repression_function(x, TRAIN_GUIDES + VAL_GUIDES, options.MIRLEN, SEQLEN, options.NUM_FEATS, options.USE_FEATS, options.PASSENGER)

    # preprocess data
    if not options.DRY_RUN:
        tpm_dataset = tpm_dataset.shuffle(buffer_size=20)
    tpm_dataset = tpm_dataset.prefetch(options.REPRESSION_BATCH_SIZE)
    tpm_train_dataset = tpm_dataset.map(_parse_fn_train, num_parallel_calls=16)
    tpm_val_dataset = tpm_dataset.map(_parse_fn_val, num_parallel_calls=16)

    # filter genes by batch
    tpm_train_dataset = tpm_train_dataset.filter(lambda x1, x2, x3, x4, x5, x6: tf.not_equal(x6, options.BATCH))
    tpm_val_dataset = tpm_val_dataset.filter(lambda x1, x2, x3, x4, x5, x6: tf.equal(x6, options.BATCH))

    # make feedable iterators
    tpm_train_iterator = tpm_train_dataset.make_initializable_iterator()
    tpm_val_iterator = tpm_val_dataset.make_initializable_iterator()

    # create handle for switching between training and validation
    tpm_handle = tf.placeholder(tf.string, shape=[])
    tpm_iterator = tf.data.Iterator.from_string_handle(tpm_handle, tpm_train_dataset.output_types)
    next_tpm_sample = parse_data_utils._build_tpm_batch(tpm_iterator, 1)
    next_tpm_batch = parse_data_utils._build_tpm_batch(tpm_iterator, options.REPRESSION_BATCH_SIZE)

    # KD data reader
    if options.PRETRAIN:
        print("Loading training KD data from")
        print(options.KD_TFRECORDS)
        kd_dataset = tf.data.TFRecordDataset(options.KD_TFRECORDS)
        kd_val_dataset = kd_dataset.take(1000)
        kd_train_dataset = kd_dataset.skip(1000)

    else:
        TRAIN_KDS_FILES = np.array([options.KD_TFRECORDS + '_{}.tfrecord'.format(mir) for mir in TRAIN_MIRS_KDS])
        print("Loading training KD data from")
        print(TRAIN_KDS_FILES)

        kd_dataset = parse_data_utils._load_multiple_tfrecords(TRAIN_KDS_FILES)

        # split into training and validation sets
        if len(VAL_MIRS_KDS) > 0:
            VAL_KDS_FILES = np.array([options.KD_TFRECORDS + '_{}.tfrecord'.format(mir) for mir in VAL_MIRS_KDS])
            print("Loading validation KD data from")
            print(VAL_KDS_FILES)
            kd_val_dataset = parse_data_utils._load_multiple_tfrecords(VAL_KDS_FILES)
            kd_train_dataset = kd_dataset

        else:
            print("Taking first 1000 kds as validation set.")
            kd_val_dataset = kd_dataset.take(1000)
            kd_train_dataset = kd_dataset.skip(1000)

    kd_train_dataset = kd_train_dataset.prefetch(options.KD_BATCH_SIZE)
    kd_val_dataset = kd_val_dataset.prefetch(1000)

    # shuffle, batch, and map datasets
    kd_train_dataset = kd_train_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=100))
    kd_train_dataset = kd_train_dataset.map(parse_data_utils._parse_log_kd_function, num_parallel_calls=16)

    # re-balance KD data towards high-affinity sites
    kd_train_dataset = kd_train_dataset.filter(parse_data_utils._filter_kds)
    kd_train_dataset = kd_train_dataset.batch(options.KD_BATCH_SIZE, drop_remainder=True)

    kd_val_dataset = kd_val_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=None)
    kd_val_dataset = kd_val_dataset.map(parse_data_utils._parse_log_kd_function)
    kd_val_dataset = kd_val_dataset.filter(parse_data_utils._filter_kds)
    kd_val_dataset = kd_val_dataset.batch(1000)

    # make feedable iterators
    kd_train_iterator = kd_train_dataset.make_initializable_iterator()
    kd_val_iterator = kd_val_dataset.make_initializable_iterator()

    # create handle for switching between training and validation
    kd_handle = tf.placeholder(tf.string, shape=[])
    kd_iterator = tf.data.Iterator.from_string_handle(kd_handle, kd_train_dataset.output_types)
    next_kd_batch_mirs, next_kd_batch_images, next_kd_batch_labels, next_kd_batch_keep_probs, next_kd_batch_stypes = kd_iterator.get_next()

    # add random sequences generator
    def gen():
        while True:
            # random_mirseq = utils.generate_random_seq(options.MIRLEN)
            # random_target = utils.get_target_no_match(random_mirseq, SEQLEN)
            # random_image = np.outer(utils.one_hot_encode(random_mirseq), utils.one_hot_encode(random_target))

            rbns1_mir = np.random.choice(TRAIN_MIRS_KDS)
            rbns1_mirseq = MIRNA_DATA.loc[rbns1_mir]['guide_seq'][:options.MIRLEN]
            rbns1_target = utils.get_target_no_match(rbns1_mirseq, SEQLEN)
            rbns1_image = np.outer(utils.one_hot_encode(rbns1_mirseq), utils.one_hot_encode(rbns1_target))

            rbns2_mir = np.random.choice(TRAIN_MIRS_KDS)
            rbns2_target = utils.generate_random_seq(3) + utils.rev_comp(MIRNA_DATA.loc[rbns2_mir]['guide_seq'][1:7]) + utils.generate_random_seq(3)
            rbns2_mirseq = utils.get_mir_no_match(rbns2_target, options.MIRLEN)
            rbns2_image = np.outer(utils.one_hot_encode(rbns2_mirseq), utils.one_hot_encode(rbns2_target))

            # yield np.array([b'random', rbns1_mir.encode('utf-8'), rbns2_mir.encode('utf-8')]), np.stack([random_image, rbns1_image, rbns2_image]), np.array([[0.0], [0.0], [0.0]]), np.array([b'no site', b'no site', b'no site'])
            yield np.array([rbns1_mir.encode('utf-8'), rbns2_mir.encode('utf-8')]), np.stack([rbns1_image, rbns2_image]), np.array([[0.0], [0.0]]), np.array([b'no site', b'no site'])

    random_seq_dataset = tf.data.Dataset.from_generator(gen, (tf.string, tf.float32, tf.float32, tf.string))
    random_seq_iterator = random_seq_dataset.make_initializable_iterator()
    random_seq_mirs, random_seq_images, random_seq_labels, random_seq_stypes = random_seq_iterator.get_next()

    NUM_EXTRA_KD_VALS = 2

    next_kd_batch = {
        'mirs': tf.concat([next_kd_batch_mirs, random_seq_mirs], axis=0),
        'images': tf.concat([next_kd_batch_images, random_seq_images], axis=0),
        'labels': tf.nn.relu(-1 * tf.concat([next_kd_batch_labels, random_seq_labels], axis=0)),
        'stypes': tf.concat([next_kd_batch_stypes, random_seq_stypes], axis=0),
    }

    # create placeholders for input data
    _dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
    # _phase_train = None
    _phase_train = tf.placeholder(tf.bool, name='phase_train')

    # build KA predictor
    _combined_x = tf.concat([next_kd_batch['images'], next_tpm_batch['images']], axis=0)
    _combined_x_4D = tf.expand_dims((_combined_x * 4.0) - 0.25, axis=3)  # reshape, zero-center input
    _pred_ka_values, _cnn_weights = model.seq2ka_predictor(
        _combined_x_4D, _dropout_rate, _phase_train,
        options.HIDDEN1, options.HIDDEN2, options.HIDDEN3, options.MIRLEN, SEQLEN
    )  # pred ka

    # make model saver
    pretrain_saver = tf.train.Saver(max_to_keep=options.NUM_EPOCHS)

    # split data into biochem and repression and get biochem loss
    _pred_biochem = _pred_ka_values[:tf.shape(next_kd_batch['images'])[0], :]
    # _ka_loss = (tf.nn.l2_loss(tf.subtract(tf.nn.relu(_pred_biochem), next_kd_batch['labels'])))# / options.KD_BATCH_SIZE
    # _ka_loss_weights = next_kd_batch['labels'] + 1
    _ka_loss = tf.nn.l2_loss(tf.nn.relu(_pred_biochem) - next_kd_batch['labels']) / 2
    _utr_ka_values = _pred_ka_values[tf.shape(next_kd_batch['images'])[0]:, :]

    # create freeAGO concentration variables
    _freeAGO_mean = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(-4.0))
    _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset', shape=[NUM_TRAIN_GUIDES], initializer=tf.constant_initializer(0.0))
    if options.PASSENGER:
        _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset', shape=[NUM_TRAIN_GUIDES], initializer=tf.constant_initializer(-1.0))
        _freeAGO_all = tf.reshape(tf.stack([_freeAGO_guide_offset, _freeAGO_pass_offset], axis=1), [-1]) + _freeAGO_mean
        _freeAGO_all_val = tf.placeholder(tf.float32, shape=[len(ALL_GUIDES) * 2], name='freeAGO_val')
    else:
        _freeAGO_all = _freeAGO_guide_offset + _freeAGO_mean
        _freeAGO_all_val = tf.placeholder(tf.float32, shape=[len(ALL_GUIDES)], name='freeAGO_val')

    # # create freeAGO concentration variables
    # # temp_slope, temp_int = -0.13517735857530447, -3.663356249809042
    # if options.PASSENGER:
    #     init_freeagos = np.array([[-3, -7.0]] *  NUM_TRAIN_GUIDES)
    #     # init_freeagos[:, 0] = (MIRNA_DATA.loc[TRAIN_GUIDES]['guide_TA'].values * temp_slope) + temp_int
    #     init_freeagos = init_freeagos.flatten()

    #     _freeAGO_all = tf.get_variable('freeAGO_all', shape=[NUM_TRAIN_GUIDES * 2],  initializer=tf.constant_initializer(init_freeagos))
    #     _freeAGO_all_val = tf.placeholder(tf.float32, shape=[len(ALL_GUIDES) * 2], name='freeAGO_val')
    #     _freeAGO_mean = tf.reduce_mean(tf.reshape(_freeAGO_all, [NUM_TRAIN_GUIDES, 2])[:, 0])
    #     _freeAGO_guide_offset = tf.reshape(_freeAGO_all, [NUM_TRAIN_GUIDES, 2])[:, 0] - _freeAGO_mean
    # else:
    #     init_freeagos = np.array([-5.0] *  NUM_TRAIN_GUIDES)
    #     _freeAGO_all = tf.get_variable('freeAGO_all', shape=[NUM_TRAIN_GUIDES],  initializer=tf.constant_initializer(init_freeagos))
    #     _freeAGO_all_val = tf.placeholder(tf.float32, shape=[len(ALL_GUIDES)], name='freeAGO_val')
    #     _freeAGO_mean = tf.reduce_mean(_freeAGO_all)
    #     _freeAGO_guide_offset = _freeAGO_all - _freeAGO_mean

    tf.summary.scalar('freeAGO', _freeAGO_mean)

    # make feature weights
    weights_init = np.array([-1.0]*options.USE_FEATS).reshape([options.USE_FEATS, 1])

    # create ts7 weight variable
    with tf.name_scope('ts7_layer'):
        with tf.name_scope('weights'):
            _ts7_weights = tf.get_variable("ts7_weights", shape=[options.USE_FEATS, 1],
                                        initializer=tf.constant_initializer(weights_init))
            tf.add_to_collection('weight', _ts7_weights)
            _decay = tf.get_variable('decay', initializer=-1.0)
            _ts7_bias = tf.get_variable('ts7_bias', initializer=1.0, trainable=False)

            model.variable_summaries(_ts7_weights)

    tf.summary.scalar('decay', _decay)

    # get logfc prediction
    _pred_logfc, _pred_logfc_normed, _repression_y_normed, _debug_item = model.get_pred_logfc_occupancy_only(
        _utr_ka_values,
        _freeAGO_all,
        next_tpm_batch,
        _ts7_weights,
        _ts7_bias,
        _decay,
        options.REPRESSION_BATCH_SIZE,
        options.PASSENGER,
        NUM_TRAIN_GUIDES,
        'pred_logfc_train',
        options.LOSS_TYPE
    )

    _pred_logfc_val, _pred_logfc_val_normed, _repression_y_val_normed, _ = model.get_pred_logfc_occupancy_only(
        _utr_ka_values,
        _freeAGO_all_val,
        next_tpm_batch,
        _ts7_weights,
        _ts7_bias,
        _decay,
        1,
        options.PASSENGER,
        len(ALL_GUIDES),
        'pred_logfc_val',
        options.LOSS_TYPE
    )

    print('pred_logfc: {}'.format(_pred_logfc))
    print(_pred_logfc_normed)
    print(_repression_y_normed)
    # _repression_weight = tf.nn.relu(-1 * _repression_y_normed) + 1.0
    _repression_loss = 2 * tf.nn.l2_loss(tf.subtract(_pred_logfc_normed, _repression_y_normed))# / (options.REPRESSION_BATCH_SIZE * NUM_TRAIN_GUIDES)

    # _offset_weight = tf.abs(tf.reduce_sum(_freeAGO_guide_offset))
    _offset_weight = tf.nn.l2_loss(_freeAGO_guide_offset)
    # _offset_weight = tf.constant(0.0)

    # define regularizer
    _weight_regularize = (
        tf.nn.l2_loss(_cnn_weights['w1']) +
        tf.nn.l2_loss(_cnn_weights['w2']) +
        tf.nn.l2_loss(_cnn_weights['w3']) +
        tf.nn.l2_loss(_cnn_weights['w4'])
    )

    tf.summary.scalar('weight_regularize', _weight_regularize)

    _regularize_term = (options.LAMBDA * (_weight_regularize))# + _offset_weight

    # define loss and train_step
    if options.PRETRAIN:
        _loss = _ka_loss
        
    else:
        _loss = (_ka_loss + _repression_loss + _regularize_term)
        # _loss = _repression_loss

    tf.summary.scalar('ka_loss', _ka_loss)
    tf.summary.scalar('repression_loss', _repression_loss)
    tf.summary.scalar('regularize_term', _regularize_term)
    tf.summary.scalar('offset_weight', _offset_weight)
    tf.summary.scalar('loss', _loss)


    _global_step = tf.Variable(0, trainable=False)
    # first_decay_steps = 1000
    # _learning_rate = tf.train.cosine_decay_restarts(options.LEARNING_RATE, _global_step, first_decay_steps)
    _learning_rate = tf.train.exponential_decay(options.LEARNING_RATE, _global_step,
                                           5000, 0.95, staircase=True)

    tf.summary.scalar('learning_rate', _learning_rate)

    _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('Update OPs:')
    print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with tf.control_dependencies(_update_ops):
        optimizer = tf.train.AdamOptimizer(_learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(loss=_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        _train_step = optimizer.apply_gradients(zip(gradients, variables), global_step=_global_step)

        for gradient, variable in zip(gradients, variables):
            tf.summary.histogram("gradients/" + variable.name, tf.nn.l2_loss(gradient))
            tf.summary.histogram("variables/" + variable.name, tf.nn.l2_loss(variable))


    # gradients = optimizer.compute_gradients(loss=_loss)
    # clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    # for gradient, variable in clipped_gradients:
    #     tf.summary.histogram("gradients/" + variable.name, tf.nn.l2_loss(gradient))
    #     tf.summary.histogram("variables/" + variable.name, tf.nn.l2_loss(variable))
    # _train_step = optimizer.apply_gradients(clipped_gradients, global_step=_global_step)

    # train_vars = tf.trainable_variables()
    # print(train_vars[0].name)
    # train_vars_no_freeAGO = [v for v in train_vars if v.name not in ['freeAGO_all:0', 'decay:0']]
    # print(len(train_vars), len(train_vars_no_freeAGO))
    # _train_step = tf.train.AdamOptimizer(_learning_rate).minimize(_loss, global_step=_global_step)
    # _train_step_no_freeAGO = tf.train.AdamOptimizer(_learning_rate).minimize(_loss, global_step=_global_step, var_list=train_vars_no_freeAGO)
    # _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(_update_ops):
    #     _train_step = tf.train.AdamOptimizer(options.LEARNING_RATE).minimize(_loss)
        # _train_step = tf.contrib.opt.AdamWOptimizer(0.001, learning_rate=options.LEARNING_RATE).minimize(_loss)

    # saver = tf.train.Saver(max_to_keep=options.NUM_EPOCHS)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=options.NUM_EPOCHS)

    logfile = open(os.path.join(options.LOGDIR, 'out.log'), 'w', -1)

    # train model
    train_losses = []
    val_losses = []
    means = []
    decays = []
    r2s = []
    lrs = []
    conf = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=24) 
    with tf.Session(config=conf) as sess:
        # Merge all the summaries and write them out
        _merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(options.LOGDIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(options.LOGDIR, 'test'))

        sess.run(tf.global_variables_initializer())
        sess.run(kd_train_iterator.initializer)
        kd_train_handle = sess.run(kd_train_iterator.string_handle())
        sess.run(random_seq_iterator.initializer)

        print(sess.run(_pred_ka_values, feed_dict={_phase_train: False, _dropout_rate: 0.0, _combined_x: test_case}))
        print(sess.run(_pred_ka_values, feed_dict={_phase_train: True, _dropout_rate: 0.0, _combined_x: test_case}))
        print(sess.run(_pred_ka_values, feed_dict={_phase_train: True, _dropout_rate: 0.5, _combined_x: test_case}))

        if options.LOAD_MODEL is not None:
            latest = tf.train.latest_checkpoint(options.LOAD_MODEL)
            print('Restoring from {}'.format(latest))
            saver.restore(sess, latest)

        print(sess.run(_pred_ka_values, feed_dict={_phase_train: False, _dropout_rate: 0.0, _combined_x: test_case}))
        print(sess.run(_pred_ka_values, feed_dict={_phase_train: True, _dropout_rate: 0.0, _combined_x: test_case}))
        print(sess.run(_pred_ka_values, feed_dict={_phase_train: True, _dropout_rate: 0.5, _combined_x: test_case}))

        num_steps = 0
        for current_epoch in range(options.NUM_EPOCHS):
            sess.run(tpm_train_iterator.initializer)
            
            # get training handle for repression data
            tpm_train_handle = sess.run(tpm_train_iterator.string_handle())

            train_feed_dict = {
                _phase_train: True,
                _dropout_rate: 0.5,
                tpm_handle: tpm_train_handle,
                kd_handle: kd_train_handle
            }

            time_start = time.time()
            while True:
                try:
                    evals = sess.run([
                        next_kd_batch,
                        next_tpm_batch,
                        _pred_biochem,
                        _pred_logfc,
                        _repression_y_normed,
                        _pred_logfc_normed,
                        _loss,
                        _ka_loss,
                        _repression_loss,
                        _regularize_term,
                        _freeAGO_mean,
                        _decay,
                        _offset_weight,
                        _utr_ka_values,
                        _debug_item,
                        _learning_rate,
                        _merged,
                        _train_step,
                    ], feed_dict=train_feed_dict)

                    if options.DRY_RUN:
                        break

                    if (num_steps % 10) == 0:
                        train_writer.add_summary(evals[-2], num_steps)
                    num_steps += 1

                except tf.errors.OutOfRangeError:
                    break

            eval_names = [
                'next_kd_batch', 'next_tpm_batch', 'ka_pred', 'pred_repression',
                'normed_tpms', 'normed_pred_repression', 'loss',
                'ka_loss', 'repression_loss', 'regularize_term',
                'freeAGO_mean', 'decay', 'offset_weight', 'utr_ka_values', 'debug', 'learning_rate', 'merged'
            ]
            evals_dict = {eval_names[ix]: evals[ix] for ix in range(len(eval_names))}

            if options.PRETRAIN:
                pretrain_saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=current_epoch)

            else:
                saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=current_epoch)

            # print('weights:')
            # temp_weights = sess.run(_cnn_weights['w3'])
            # print(np.min(temp_weights), np.max(temp_weights), np.mean(temp_weights), np.std(temp_weights))

            # plot weights
            conv_weights = sess.run(_cnn_weights['w1'])
            xlabels = ['A', 'U', 'C', 'G']
            ylabels = ['A', 'U', 'C', 'G']
            utils.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1.pdf'))

            # plot importance matrix
            conv_weights = np.abs(sess.run(_cnn_weights['w3']))
            conv_weights = np.sum(conv_weights, axis=(2, 3))
            vmin, vmax = np.min(conv_weights), np.max(conv_weights)
            xlabels = ['s{}'.format(i + 1) for i in range(SEQLEN)]
            ylabels = ['m{}'.format(i + 1) for i in list(range(options.MIRLEN))[::-1]]
            fig = plt.figure(figsize=(4, 4))
            sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                        cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
            plt.savefig(os.path.join(options.LOGDIR, 'convolution3.pdf'))
            plt.close()

            colors = [STYPE_COLOR_DICT[x] for x in evals_dict['next_kd_batch']['stypes']]
            fig = plt.figure(figsize=(7, 7))
            plt.scatter(evals_dict['ka_pred'].flatten(), evals_dict['next_kd_batch']['labels'].flatten(), color=colors)
            plt.savefig(os.path.join(options.LOGDIR, 'train_ka_scatter.png'))
            plt.close()

            fig = plt.figure(figsize=(7, 7))
            plt.scatter(evals_dict['normed_pred_repression'].flatten(), evals_dict['normed_tpms'].flatten())
            plt.savefig(os.path.join(options.LOGDIR, 'train_repression_scatter.png'))
            plt.close()

            train_losses.append(evals_dict['loss'])
            fig = plt.figure(figsize=(7, 5))
            plt.plot(train_losses)
            plt.savefig(os.path.join(options.LOGDIR, 'train_losses.png'))
            plt.close()

            lrs.append(evals_dict['learning_rate'])
            fig = plt.figure(figsize=(7, 5))
            plt.plot(lrs)
            plt.savefig(os.path.join(options.LOGDIR, 'learning_rates.png'))
            plt.close()

            means.append(evals_dict['freeAGO_mean'])
            fig = plt.figure(figsize=(7, 5))
            plt.plot(means)
            plt.savefig(os.path.join(options.LOGDIR, 'train_freeAGO_means.png'))
            plt.close()

            decays.append(evals_dict['decay'])
            fig = plt.figure(figsize=(7, 5))
            plt.plot(decays)
            plt.savefig(os.path.join(options.LOGDIR, 'decays.png'))
            plt.close()

            # logfile.write('Time for epoch: {}\n'.format(time.time() - time_start))
            logfile.write('Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(current_epoch, evals_dict['loss'], evals_dict['ka_loss'], evals_dict['repression_loss'], evals_dict['regularize_term']))
            print('Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(current_epoch, evals_dict['loss'], evals_dict['ka_loss'], evals_dict['repression_loss'], evals_dict['regularize_term']))
            print(sess.run(_ts7_weights).flatten())
            print(sess.run(_decay))
            print(evals_dict['offset_weight'])

            if options.PASSENGER:
                current_freeAGO_all = sess.run(_freeAGO_all).reshape([-1, 2])
            else:
                current_freeAGO_all = sess.run(_freeAGO_all).reshape([-1, 1])

            print(current_freeAGO_all)

            train_guide_tas = MIRNA_DATA_USE_TPMS.loc[TRAIN_GUIDES]['guide_TA'].values
            new_freeago = []

            fig = plt.figure(figsize=(7, 7))
            plt.scatter(train_guide_tas, current_freeAGO_all[:, 0])

            if len(VAL_GUIDES) > 0:
                slope, inter = stats.linregress(train_guide_tas, current_freeAGO_all[:, 0])[:2]
                val_guide_ta = MIRNA_DATA_USE_TPMS.loc[VAL_GUIDES[0]]['guide_TA']
                new_freeago.append(slope * val_guide_ta + inter)
                plt.scatter([val_guide_ta], new_freeago[0], color='red')

            plt.savefig(os.path.join(options.LOGDIR, 'train_ta_guide_freeago.png'))
            plt.close()

            if options.PASSENGER:
                train_pass_tas = MIRNA_DATA_USE_TPMS.loc[TRAIN_GUIDES]['pass_TA'].values

                fig = plt.figure(figsize=(7, 7))
                plt.scatter(train_pass_tas, current_freeAGO_all[:, 1])

                if len(VAL_GUIDES) > 0:
                    slope, inter = stats.linregress(train_pass_tas, current_freeAGO_all[:, 1])[:2]
                    val_pass_ta = MIRNA_DATA_USE_TPMS.loc[VAL_GUIDES[0]]['pass_TA']
                    new_freeago.append(slope * val_pass_ta + inter)
                    plt.scatter([val_pass_ta], new_freeago[1], color='red')

                plt.savefig(os.path.join(options.LOGDIR, 'train_ta_pass_freeago.png'))
                plt.close()

            if len(VAL_GUIDES) > 0:
                current_freeAGO_all_val = np.concatenate([current_freeAGO_all, np.array([new_freeago])], axis=0).flatten()
            else:
                current_freeAGO_all_val = current_freeAGO_all.flatten()

            sess.run(tpm_val_iterator.initializer)
            tpm_val_handle = sess.run(tpm_val_iterator.string_handle())
            pred_vals, real_vals, real_nsites = [], [], []
            current_transcripts = []
            while True:
                try:
                    temp_tpm_batch = sess.run(next_tpm_sample, feed_dict={tpm_handle: tpm_val_handle})
                    temp_nsites = temp_tpm_batch['nsites']
                    current_transcripts += list(temp_tpm_batch['transcripts'].flatten())
                    if options.PASSENGER:
                        real_nsites.append(temp_nsites.reshape([-1, len(ALL_GUIDES), 2])[:, :, 0])
                    else:
                        real_nsites.append(temp_nsites.reshape([-1, len(ALL_GUIDES)]))
                    real_vals.append(temp_tpm_batch['labels'])
                    ka_vals = sess.run(_pred_ka_values, feed_dict={_phase_train: False, _dropout_rate: 0.0, _combined_x: temp_tpm_batch['images']})
                    pred_vals.append(sess.run(_pred_logfc_val,
                        feed_dict={
                            _utr_ka_values: ka_vals,
                            next_tpm_batch['nsites']: temp_tpm_batch['nsites'],
                            next_tpm_batch['features']: temp_tpm_batch['features'],
                            next_tpm_batch['labels']: temp_tpm_batch['labels'],
                            _freeAGO_all_val: current_freeAGO_all_val
                        }))
                except tf.errors.OutOfRangeError:
                    break

            real_nsites = np.concatenate(real_nsites)
            pred_vals = np.concatenate(pred_vals)
            real_vals = np.concatenate(real_vals)

            print('val shape:')
            print(real_vals.shape)

            fig = plt.figure(figsize=(7, 7))
            plt.scatter(real_nsites[:, -1], pred_vals[:, -1])
            plt.savefig(os.path.join(options.LOGDIR, 'val_nsites_scatter.png'))
            plt.close()

            pred_vals_normed = pred_vals - np.mean(pred_vals, axis=1).reshape([-1,1])
            real_vals_normed = real_vals - np.mean(real_vals, axis=1).reshape([-1,1])

            pred_df = pd.DataFrame({
                'transcript': np.repeat(current_transcripts, len(ALL_GUIDES)),
                'batch': options.BATCH,
                'mir': (TRAIN_GUIDES + VAL_GUIDES) * len(current_transcripts),
                'pred': pred_vals.flatten(),
                'label': real_vals.flatten(),
                'pred_normed': pred_vals_normed.flatten(),
                'label_normed': real_vals_normed.flatten()
            })

            pred_df.to_csv(os.path.join(options.LOGDIR, 'pred_df.txt'), sep='\t', index=False)

            val_losses.append(np.sum(np.square(pred_vals_normed[:, -1] - real_vals_normed[:, -1])))
            fig = plt.figure(figsize=(7, 5))
            plt.plot(val_losses)
            plt.savefig(os.path.join(options.LOGDIR, 'val_losses.png'))
            plt.close()

            r2s.append(stats.linregress(pred_vals_normed[:, -1], real_vals_normed[:, -1])[2]**2)
            fig = plt.figure(figsize=(7, 5))
            plt.plot(r2s)
            plt.savefig(os.path.join(options.LOGDIR, 'val_r2s.png'))
            plt.close()

            fig = plt.figure(figsize=(7, 7))
            plt.scatter(pred_vals_normed[:, -1], real_vals_normed[:, -1], c=real_nsites[:, -1])
            plt.savefig(os.path.join(options.LOGDIR, 'val_tpm_scatter.png'))
            plt.close()

            fig = plt.figure(figsize=(7, 7))
            plt.scatter(pred_vals_normed.flatten(), real_vals_normed.flatten(), c=real_nsites.flatten(), alpha=0.5)
            plt.savefig(os.path.join(options.LOGDIR, 'all_tpm_scatter.png'))
            plt.close()

            sess.run(kd_val_iterator.initializer)
            kd_val_handle = sess.run(kd_val_iterator.string_handle())
            temp_kd_batch = sess.run(next_kd_batch, feed_dict={kd_handle: kd_val_handle})
            ka_vals = sess.run(_pred_ka_values, feed_dict={_phase_train: False, _dropout_rate: 0.0, _combined_x: temp_kd_batch['images']})

            fig = plt.figure(figsize=(7, 7))
            colors = [STYPE_COLOR_DICT[x] for x in temp_kd_batch['stypes']]
            plt.scatter(ka_vals.flatten(), temp_kd_batch['labels'].flatten(), color=colors)
            plt.savefig(os.path.join(options.LOGDIR, 'val_ka_scatter.png'))
            plt.close()
            
            print(sess.run(_pred_ka_values, feed_dict={_phase_train: False, _dropout_rate: 0.0, _combined_x: test_case}))
            print(sess.run(_pred_ka_values, feed_dict={_phase_train: True, _dropout_rate: 0.0, _combined_x: test_case}))
            print(sess.run(_pred_ka_values, feed_dict={_phase_train: True, _dropout_rate: 0.5, _combined_x: test_case}))

            if options.DRY_RUN:
                print(evals_dict['next_tpm_batch']['images'].shape)
                print(evals_dict['next_kd_batch']['images'].shape)
                print(test_case.shape)
                print(evals_dict['next_tpm_batch']['transcripts'])
                print(evals_dict['next_tpm_batch']['nsites'])
                print(evals_dict['next_tpm_batch']['labels'])
                print(evals_dict['next_tpm_batch']['batches'])
                print(evals_dict['utr_ka_values'][-1])
                print(np.max(evals_dict['next_tpm_batch']['images'][-1, :, :], axis=0).reshape([-1, 4]))
                print(np.max(evals_dict['next_tpm_batch']['images'][-1, :, :], axis=1).reshape([-1, 4]))
                # print(evals_dict['next_tpm_batch']['transcripts'])
                # temp_nsites = evals_dict['next_tpm_batch']['nsites']
                # temp_pred_ka = evals_dict['utr_ka_values'].flatten()
                # temp_is_orf = evals_dict['next_tpm_batch']['features'].flatten()
                # print(evals_dict['pred_repression'])
                # temp_total = 0
                # for ix, n in enumerate(temp_nsites):
                #     print(n, temp_pred_ka[temp_total: temp_total + n], temp_is_orf[temp_total: temp_total + n])
                #     temp_total += n

                # print(evals_dict['pred_repression'], evals_dict['normed_pred_repression'])
                # print(evals_dict['next_tpm_batch']['labels'], evals_dict['normed_tpms'])
                # A = evals_dict['next_kd_batch']['labels']
                # B = np.maximum(0, evals_dict['ka_pred'])
                # print(A, B)
                # print(np.sum(np.square(A-B)))
                # print(evals_dict['ka_loss'] * 2)

                # while True:
                #     temp_kd_batch = sess.run(next_kd_batch, feed_dict={kd_handle: kd_train_handle})
                #     if b'8mer' in temp_kd_batch['stypes']:
                #         break

                # vals1 = sess.run(_pred_ka_values, feed_dict={_phase_train: False, _dropout_rate: 0.0, _combined_x: temp_kd_batch['images']}).flatten()
                # vals2 = sess.run(_pred_ka_values, feed_dict={_phase_train: True, _dropout_rate: 0.0, _combined_x: temp_kd_batch['images']}).flatten()
                # vals3 = sess.run(_pred_ka_values, feed_dict={_phase_train: True, _dropout_rate: 0.5, _combined_x: temp_kd_batch['images']}).flatten()

                # ix = 0
                # for blah in zip(temp_kd_batch['mirs'], temp_kd_batch['stypes'], temp_kd_batch['labels'].flatten(), vals1, vals2, vals3):
                #     print(blah)
                #     if blah[1] == b'8mer':
                #         print(np.max(temp_kd_batch['images'][ix, :, :], axis=0).reshape([-1, 4]).astype(int))
                #         print(np.max(temp_kd_batch['images'][ix, :, :], axis=1).reshape([-1, 4]).astype(int))
                #         print(temp_kd_batch['images'][ix, :, :])
                #     ix += 1

                
                
                break


