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
    parser.add_option("--lambda1", dest="LAMBDA1", help="regularizer weight", type=float)
    parser.add_option("--lambda2", dest="LAMBDA2", help="offset weight", type=float)
    parser.add_option("--repression_weight", dest="REPRESSION_WEIGHT", help="regularizer weight", type=float)
    parser.add_option("--lr", dest="LEARNING_RATE", help="starting learning rate", type=float)
    parser.add_option("--logdir", dest="LOGDIR", help="directory for writing logs", default=None)
    parser.add_option("--load_model", dest="LOAD_MODEL", help="if supplied, load latest model from this directory", default=None)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--dry_run", dest="DRY_RUN", help="if true, do dry run", default=False, action='store_true')
    parser.add_option("--pretrain", dest="PRETRAIN", help="if true, do pretraining step", default=False, action='store_true')

    (options, args) = parser.parse_args()

    tf.reset_default_graph()

    if options.LOGDIR is not None:
        SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

        if (not os.path.isdir(options.LOGDIR)):
            os.makedirs(options.LOGDIR)

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
    kd_val_dataset = kd_val_dataset.batch(1000, drop_remainder=False)

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
            random_mirseq = utils.generate_random_seq(options.MIRLEN)
            random_target = utils.get_target_no_match(random_mirseq, SEQLEN)
            random_image = np.outer(utils.one_hot_encode(random_mirseq), utils.one_hot_encode(random_target))

            rbns1_mir = np.random.choice(TRAIN_MIRS_KDS)
            rbns1_mirseq = MIRNA_DATA.loc[rbns1_mir]['guide_seq'][:options.MIRLEN]
            rbns1_target = utils.get_target_no_match(rbns1_mirseq, SEQLEN)
            rbns1_image = np.outer(utils.one_hot_encode(rbns1_mirseq), utils.one_hot_encode(rbns1_target))

            rbns2_mir = np.random.choice(TRAIN_MIRS_KDS)
            rbns2_target = utils.generate_random_seq(3) + utils.rev_comp(MIRNA_DATA.loc[rbns2_mir]['guide_seq'][1:7]) + utils.generate_random_seq(3)
            rbns2_mirseq = utils.get_mir_no_match(rbns2_target, options.MIRLEN)
            rbns2_image = np.outer(utils.one_hot_encode(rbns2_mirseq), utils.one_hot_encode(rbns2_target))

            yield np.array([b'random', rbns1_mir.encode('utf-8'), rbns2_mir.encode('utf-8')]), np.stack([random_image, rbns1_image, rbns2_image]), np.array([[0.0], [0.0], [0.0]]), np.array([b'no site', b'no site', b'no site'])
            # yield np.array([rbns1_mir.encode('utf-8'), rbns2_mir.encode('utf-8')]), np.stack([rbns1_image, rbns2_image]), np.array([[0.0], [0.0]]), np.array([b'no site', b'no site'])

    random_seq_dataset = tf.data.Dataset.from_generator(gen, (tf.string, tf.float32, tf.float32, tf.string))
    random_seq_iterator = random_seq_dataset.make_initializable_iterator()
    random_seq_mirs, random_seq_images, random_seq_labels, random_seq_stypes = random_seq_iterator.get_next()

    NUM_EXTRA_KD_VALS = 3

    next_kd_batch = {
        'mirs': tf.concat([next_kd_batch_mirs, random_seq_mirs], axis=0),
        'images': tf.concat([next_kd_batch_images, random_seq_images], axis=0),
        'labels': tf.nn.relu(-1 * tf.concat([next_kd_batch_labels, random_seq_labels], axis=0)),
        'stypes': tf.concat([next_kd_batch_stypes, random_seq_stypes], axis=0),
    }

    # create placeholders for input data
    _dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
    _phase_train = tf.placeholder(tf.bool, name='phase_train')

    # build KA predictor
    _combined_x = tf.concat([next_kd_batch['images'], next_tpm_batch['images']], axis=0)
    _combined_x_4D = tf.expand_dims((_combined_x * 4.0) - 0.25, axis=3)  # reshape, zero-center input
    _pred_ka_values, _cnn_weights = model.seq2ka_predictor(
        _combined_x_4D, _dropout_rate, _phase_train,
        options.HIDDEN1, options.HIDDEN2, options.HIDDEN3, options.MIRLEN, SEQLEN
    )  # pred ka

    # make model saver
    pretrain_saver = tf.train.Saver(tf.global_variables(), max_to_keep=options.NUM_EPOCHS)

    # split data into biochem and repression and get biochem loss
    _pred_biochem = _pred_ka_values[:tf.shape(next_kd_batch['images'])[0], :]
    _ka_loss = tf.nn.l2_loss(tf.nn.relu(_pred_biochem) - next_kd_batch['labels']) / options.KD_BATCH_SIZE
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

    # # get logfc prediction
    # _pred_logfc, _pred_logfc_normed, _repression_y_normed, _debug_item = model.get_pred_logfc_occupancy_only(
    #     _utr_ka_values,
    #     _freeAGO_all,
    #     next_tpm_batch,
    #     _ts7_weights,
    #     _ts7_bias,
    #     _decay,
    #     options.REPRESSION_BATCH_SIZE,
    #     options.PASSENGER,
    #     NUM_TRAIN_GUIDES,
    #     'pred_logfc_train',
    #     options.LOSS_TYPE
    # )

    _pred_logfc2, _pred_logfc_normed2, _repression_y_normed2, _debug_item2 = model.get_pred_logfc_occupancy_only2(
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

    conf = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=24) 
    with tf.Session(config=conf) as sess:
        # writer = tf.summary.FileWriter("tensorboard_test", sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(kd_train_iterator.initializer)
        kd_train_handle = sess.run(kd_train_iterator.string_handle())
        sess.run(random_seq_iterator.initializer)

        sess.run(tpm_train_iterator.initializer)

        print(sess.run(_freeAGO_all))
            
        # get training handle for repression data
        tpm_train_handle = sess.run(tpm_train_iterator.string_handle())

        train_feed_dict = {
            _phase_train: True,
            _dropout_rate: 0.5,
            tpm_handle: tpm_train_handle,
            kd_handle: kd_train_handle
        }

        print(sess.run(_debug_item2, feed_dict=train_feed_dict))

        # t0 = time.time()
        # for _ in range(100):
        #     results = sess.run(_debug_item2, feed_dict=train_feed_dict)

        # print(time.time() - t0)

