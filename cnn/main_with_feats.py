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
    parser.add_option("--batch", dest="BATCH", help="which batch to withold from training", type=int, default=None)
    parser.add_option("--loss_type", dest="LOSS_TYPE", help="which loss strategy")
    parser.add_option("--dropout_rate", dest="DROPOUT_RATE", help="dropout_rate", type=float)
    parser.add_option("--lambda1", dest="LAMBDA1", help="regularizer weight", type=float)
    parser.add_option("--lambda2", dest="LAMBDA2", help="offset weight", type=float)
    parser.add_option("--repression_weight", dest="REPRESSION_WEIGHT", help="regularizer weight", type=float)
    parser.add_option("--lr", dest="LEARNING_RATE", help="starting learning rate", type=float)
    parser.add_option("--logdir", dest="LOGDIR", help="directory for writing logs", default=None)
    parser.add_option("--load_model", dest="LOAD_MODEL", help="if supplied, load latest model from this directory", default=None)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--fix_layers", dest="FIX_LAYERS", help="if true, fix earlier layers", default=False, action='store_true')
    parser.add_option("--dry_run", dest="DRY_RUN", help="if true, do dry run", default=False, action='store_true')
    parser.add_option("--pretrain", dest="PRETRAIN", help="if true, do pretraining step", default=False, action='store_true')

    (options, args) = parser.parse_args()

    tf.reset_default_graph()

    testcase_mirseq = utils.generate_random_seq(options.MIRLEN)
    testcase_target = 'AA' + utils.rev_comp(testcase_mirseq[1:8]) + 'AAA'
    testcase_image = np.outer(utils.one_hot_encode(testcase_mirseq), utils.one_hot_encode(testcase_target))
    testcase_image = np.expand_dims(testcase_image, 0)

    if options.LOGDIR is not None:
        # make log directory if it doesn't exist
        if (not os.path.isdir(options.LOGDIR)):
            os.makedirs(options.LOGDIR)

        # paths for saved models
        PRETRAIN_SAVE_PATH = os.path.join(options.LOGDIR, 'pretrain_saved')
        SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

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

    # split repression miRNAs into training and testing
    if options.VAL_MIR == 'none':
        TRAIN_GUIDES = [x for x in ALL_GUIDES if x != 'mir204'] + ['mir204']
        VAL_GUIDES = []
    else:
        if options.VAL_MIR not in ALL_GUIDES:
            raise ValueError('Test miRNA not in mirseqs file.')
        TRAIN_GUIDES = [m for m in ALL_GUIDES if m != options.VAL_MIR]
        VAL_GUIDES = [options.VAL_MIR]

    # split RBNS miRNAs into training and testing
    if options.VAL_MIR in list(MIRNA_DATA_WITH_RBNS.index):
        TRAIN_MIRS_KDS = [x for x in list(MIRNA_DATA_WITH_RBNS.index) if x != options.VAL_MIR]
        VAL_MIRS_KDS = [options.VAL_MIR]
    elif options.VAL_MIR == 'none':
        TRAIN_MIRS_KDS = list(MIRNA_DATA_WITH_RBNS.index)
        VAL_MIRS_KDS = ['mir204']
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
    if options.BATCH is not None:
        tpm_train_dataset = tpm_train_dataset.filter(lambda x1, x2, x3, x4, x5, x6: tf.not_equal(x6, options.BATCH))
        tpm_val_dataset = tpm_val_dataset.filter(lambda x1, x2, x3, x4, x5, x6: tf.equal(x6, options.BATCH))
    # else:
    #     tpm_val_dataset = tpm_val_dataset.filter(lambda x1, x2, x3, x4, x5, x6: tf.equal(x6, 3))

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
    kd_train_dataset = kd_train_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
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

    # augment data with random sequences generator
    NUM_EXTRA_KD_VALS = 3
    def gen():
        while True:
            random_mirs, random_images, random_labels, random_stypes = [], [], [], []

            # choose one of the RBNS miRNAs, generate target with no pairing, and assign KD of 0
            rbns1_mir = np.random.choice(TRAIN_MIRS_KDS)
            random_mirs.append(rbns1_mir.encode('utf-8'))
            rbns1_mirseq = MIRNA_DATA.loc[rbns1_mir]['guide_seq'][:options.MIRLEN]
            rbns1_target = utils.get_target_no_match(rbns1_mirseq, SEQLEN)
            random_images.append(np.outer(utils.one_hot_encode(rbns1_mirseq), utils.one_hot_encode(rbns1_target)))
            random_labels.append([0.0])
            random_stypes.append(b'no site')

            # generate random 8mer pair and assign KD of average 8mer
            random_mirseq = utils.generate_random_seq(options.MIRLEN)
            # random_target = utils.get_target_no_match(random_mirseq, SEQLEN)
            random_mirs.append(b'random')
            up_flank = utils.generate_random_seq(2)
            down_flank = utils.generate_random_seq(2)
            random_target = up_flank + utils.rev_comp(random_mirseq[1:8]) + 'A' + down_flank
            random_images.append(np.outer(utils.one_hot_encode(random_mirseq), utils.one_hot_encode(random_target)))
            flank_ATs = np.sum([x in ['A','T'] for x in (up_flank + down_flank)])
            random_labels.append([-4.5 - (0.5 * flank_ATs)])
            random_stypes.append(b'8mer')

            # generate miRNA and target with no pairing and assign KD of 0
            rbns2_mir = np.random.choice(TRAIN_MIRS_KDS)
            random_mirs.append(rbns2_mir.encode('utf-8'))
            rbns2_target = utils.generate_random_seq(3) + utils.rev_comp(MIRNA_DATA.loc[rbns2_mir]['guide_seq'][1:7]) + utils.generate_random_seq(3)
            rbns2_mirseq = utils.get_mir_no_match(rbns2_target, options.MIRLEN)
            random_images.append(np.outer(utils.one_hot_encode(rbns2_mirseq), utils.one_hot_encode(rbns2_target)))
            random_labels.append([0.0])
            random_stypes.append(b'no site')

            yield np.array(random_mirs), np.stack(random_images), np.array(random_labels), np.array(random_stypes)

    random_seq_dataset = tf.data.Dataset.from_generator(gen, (tf.string, tf.float32, tf.float32, tf.string))
    random_seq_iterator = random_seq_dataset.make_initializable_iterator()
    random_seq_mirs, random_seq_images, random_seq_labels, random_seq_stypes = random_seq_iterator.get_next()

    _global_step = tf.Variable(0, trainable=False)
    tf.summary.scalar('global_step', _global_step)

    _epoch_num = tf.Variable(0, trainable=False, dtype=tf.float32)
    _increment_epoch_op = tf.assign(_epoch_num, _epoch_num + 1.0)

    # # decaying weights
    # _extra_weight = tf.constant([[1.0]] * NUM_EXTRA_KD_VALS) * (1.0 / (tf.pow(tf.cast(_global_step, tf.float32)/200.0, 0.75) + 1))
    # _kd_weights = tf.concat([tf.constant([[1.0]] * options.KD_BATCH_SIZE), _extra_weight], axis=0)

    # linearly decaying weights
    # _kd_decay = tf.nn.relu(1.0 - (_epoch_num / 25.0))
    # tf.summary.scalar('kd_decay', _kd_decay)
    # _kd_weights = _kd_decay * tf.constant(([[1.0]] * options.KD_BATCH_SIZE) + ([[0.1]] * NUM_EXTRA_KD_VALS))

    _kd_weights = tf.constant(([[1.0]] * options.KD_BATCH_SIZE) + ([[0.1]] * NUM_EXTRA_KD_VALS))
    

    next_kd_batch = {
        'mirs': tf.concat([next_kd_batch_mirs, random_seq_mirs], axis=0),
        'images': tf.concat([next_kd_batch_images, random_seq_images], axis=0),
        'labels': tf.nn.relu(-1 * tf.concat([next_kd_batch_labels, random_seq_labels], axis=0)),
        'stypes': tf.concat([next_kd_batch_stypes, random_seq_stypes], axis=0)
    }

    # create placeholders for input data
    _dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
    _phase_train = tf.placeholder(tf.bool, name='phase_train')

    # build KA predictor
    _combined_x = tf.concat([next_kd_batch['images'], next_tpm_batch['images']], axis=0, name='combined_x')
    _combined_x_4D = tf.expand_dims((_combined_x * 4.0) - 0.25, axis=3)  # reshape, zero-center input
    _pred_ka_values, _cnn_weights = model.seq2ka_predictor(
        _combined_x_4D, _dropout_rate, _phase_train,
        options.HIDDEN1, options.HIDDEN2, options.HIDDEN3, options.MIRLEN, SEQLEN
    )

    # make KD model saver
    pretrain_saver = tf.train.Saver(tf.global_variables(), max_to_keep=options.NUM_EPOCHS)

    # split data into biochem and repression and get biochem loss
    _pred_biochem = _pred_ka_values[:tf.shape(next_kd_batch['images'])[0], :]
    _ka_weighted_difference = _kd_weights * tf.square(tf.nn.relu(_pred_biochem) - next_kd_batch['labels'])
    _ka_loss = tf.reduce_sum(_ka_weighted_difference) / (2 * options.KD_BATCH_SIZE)
    # _ka_loss = tf.nn.l2_loss(tf.nn.relu(_pred_biochem) - next_kd_batch['labels']) / options.KD_BATCH_SIZE
    _utr_ka_values = _pred_ka_values[tf.shape(next_kd_batch['images'])[0]:, :]

    # create freeAGO concentration variables
    if options.FIX_LAYERS:
        FREEAGO_INIT = -4.9
        FREEAGO_GUIDE_OFFSET = 0.0
        FREEAGO_PASS_OFFSET = -2.0
    else:
        FREEAGO_INIT = -4.0
        FREEAGO_GUIDE_OFFSET = 0.0
        FREEAGO_PASS_OFFSET = -1.0
        

    _freeAGO_mean = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(FREEAGO_INIT))
    _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset', shape=[NUM_TRAIN_GUIDES], initializer=tf.constant_initializer(FREEAGO_GUIDE_OFFSET))
    if options.PASSENGER:
        _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset', shape=[NUM_TRAIN_GUIDES], initializer=tf.constant_initializer(FREEAGO_PASS_OFFSET))
        _freeAGO_all = tf.reshape(tf.stack([_freeAGO_guide_offset, _freeAGO_pass_offset], axis=1), [-1]) + _freeAGO_mean
        _freeAGO_all_val = tf.placeholder(tf.float32, shape=[len(ALL_GUIDES) * 2], name='freeAGO_val')
    else:
        _freeAGO_all = _freeAGO_guide_offset + _freeAGO_mean
        _freeAGO_all_val = tf.placeholder(tf.float32, shape=[len(ALL_GUIDES)], name='freeAGO_val')

    tf.summary.scalar('freeAGO', _freeAGO_mean)

    # make feature weights
    if options.FIX_LAYERS:
        WEIGHTS_INIT = np.array([-2.22] + [-0.1]*(options.USE_FEATS - 1)).reshape([options.USE_FEATS, 1])
        DECAY_INIT = 0.0
    else:
        WEIGHTS_INIT = np.array([-0.2]*options.USE_FEATS).reshape([options.USE_FEATS, 1])
        DECAY_INIT = -1.5

    # create ts7 weight variable
    with tf.name_scope('ts7_layer'):
        with tf.name_scope('weights'):
            _ts7_weights = tf.get_variable("ts7_weights", shape=[options.USE_FEATS, 1],
                                        initializer=tf.constant_initializer(WEIGHTS_INIT))
            tf.add_to_collection('weight', _ts7_weights)
            _decay = tf.get_variable('decay', initializer=DECAY_INIT)
            _ts7_bias = tf.get_variable('ts7_bias', initializer=0.0, trainable=False)
            model.variable_summaries(_ts7_weights)

    tf.summary.scalar('decay', _decay)
    tf.summary.scalar('ts7_bias', _ts7_bias)

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

    _repression_loss = options.REPRESSION_WEIGHT * tf.nn.l2_loss(tf.subtract(_pred_logfc_normed, _repression_y_normed)) / (options.REPRESSION_BATCH_SIZE)

    # define regularizers
    _offset_weight = tf.nn.l2_loss(_freeAGO_guide_offset)
    _weight_regularize = (
        tf.nn.l2_loss(_cnn_weights['w1']) +
        tf.nn.l2_loss(_cnn_weights['w2']) +
        tf.nn.l2_loss(_cnn_weights['w3']) +
        tf.nn.l2_loss(_cnn_weights['w4'])
    )
    _regularize_term = (options.LAMBDA1 * _weight_regularize) + (options.LAMBDA2 * _offset_weight)

    # add to tensorboard
    tf.summary.scalar('offset_weight', _offset_weight)
    tf.summary.scalar('weight_regularize', _weight_regularize)
    tf.summary.scalar('regularize_term', _regularize_term)

    # define loss
    if options.PRETRAIN:
        _loss = _ka_loss
        
    else:
        _loss = (_ka_loss + _repression_loss + _regularize_term)

    # add to tensorboard
    tf.summary.scalar('ka_loss', _ka_loss)
    tf.summary.scalar('repression_loss', _repression_loss)
    tf.summary.scalar('loss', _loss)

    # add rate decay
    # _learning_rate = tf.train.exponential_decay(options.LEARNING_RATE, _global_step,
    #                                        5000, 0.90, staircase=True)

    _learning_rate = tf.train.cosine_decay(options.LEARNING_RATE, _global_step, 30000)

    # # add SGDR decay
    # _learning_rate = tf.train.cosine_decay_restarts(options.LEARNING_RATE, _global_step, 2000, t_mul=2.0, m_mul=0.75, alpha=0.0)
    tf.summary.scalar('learning_rate', _learning_rate)

    _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('Update OPs:')
    print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with tf.control_dependencies(_update_ops):
        optimizer = tf.train.AdamOptimizer(_learning_rate)

        # get gradients, fix earlier layers if training more features
        if options.FIX_LAYERS:
            var_list = [_freeAGO_mean, _freeAGO_guide_offset, _ts7_weights, _decay]
            if options.PASSENGER:
                var_list.append(_freeAGO_pass_offset)
            gradients, variables = zip(*optimizer.compute_gradients(loss=_loss, var_list=var_list))
        else:
            gradients, variables = zip(*optimizer.compute_gradients(loss=_loss))

        # clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        _train_step = optimizer.apply_gradients(zip(gradients, variables), global_step=_global_step)

        for gradient, variable in zip(gradients, variables):
            tf.summary.histogram("gradients/" + variable.name, tf.nn.l2_loss(gradient))
            tf.summary.histogram("variables/" + variable.name, tf.nn.l2_loss(variable))

    # make saver for whole model
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=options.NUM_EPOCHS)

    if options.LOGDIR is not None:
        logfile = open(os.path.join(options.LOGDIR, 'out.log'), 'wb', buffering=0)

    # train model
    train_losses = []
    val_losses = []
    means = []
    decays = []
    r2s = []
    lrs = []
    USE_BATCH_NORM = (options.FIX_LAYERS is False)
    conf = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=24) 
    with tf.Session(config=conf) as sess:

        # Merge all the summaries and write them out
        _merged = tf.summary.merge_all()
        if options.LOGDIR is not None:
            train_writer = tf.summary.FileWriter(os.path.join(options.LOGDIR, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(options.LOGDIR, 'test'))

        sess.run(tf.global_variables_initializer())
        sess.run(kd_train_iterator.initializer)
        kd_train_handle = sess.run(kd_train_iterator.string_handle())
        sess.run(random_seq_iterator.initializer)

        print(sess.run(_pred_ka_values, feed_dict={_phase_train: False, _dropout_rate: 0.0, _combined_x: testcase_image}))
        print(sess.run(_pred_ka_values, feed_dict={_phase_train: USE_BATCH_NORM, _dropout_rate: 0.0, _combined_x: testcase_image}))
        print(sess.run(_pred_ka_values, feed_dict={_phase_train: USE_BATCH_NORM, _dropout_rate: options.DROPOUT_RATE, _combined_x: testcase_image}))

        if options.LOAD_MODEL is not None:
            latest = tf.train.latest_checkpoint(options.LOAD_MODEL)
            print('Restoring from {}'.format(latest))
            saver.restore(sess, latest)

            sess.run(tf.variables_initializer([_global_step]))

        print(sess.run(_pred_ka_values, feed_dict={_phase_train: False, _dropout_rate: 0.0, _combined_x: testcase_image}))
        print(sess.run(_pred_ka_values, feed_dict={_phase_train: USE_BATCH_NORM, _dropout_rate: 0.0, _combined_x: testcase_image}))
        print(sess.run(_pred_ka_values, feed_dict={_phase_train: USE_BATCH_NORM, _dropout_rate: options.DROPOUT_RATE, _combined_x: testcase_image}))

        num_steps = 0
        for current_epoch in range(options.NUM_EPOCHS):
            sess.run(tpm_train_iterator.initializer)
            
            # get training handle for repression data
            tpm_train_handle = sess.run(tpm_train_iterator.string_handle())

            train_feed_dict = {
                _phase_train: USE_BATCH_NORM,
                _dropout_rate: options.DROPOUT_RATE,
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
                        _kd_weights,
                        _merged,
                        _train_step,
                    ], feed_dict=train_feed_dict)

                    if options.DRY_RUN:
                        break

                    # every 10 steps add to tensorboard
                    if options.LOGDIR is not None:
                        if (num_steps % 10) == 0:
                            train_writer.add_summary(evals[-2], num_steps)
                    num_steps += 1

                except tf.errors.OutOfRangeError:
                    break

            sess.run(_increment_epoch_op)

            # evaluate on validation set
            if options.PASSENGER:
                current_freeAGO_all = sess.run(_freeAGO_all).reshape([-1, 2])
            else:
                current_freeAGO_all = sess.run(_freeAGO_all).reshape([-1, 1])

            # infer freeAGO of validation miRNA from it's target abundance
            train_guide_tas = MIRNA_DATA_USE_TPMS.loc[TRAIN_GUIDES]['guide_TA'].values
            new_freeago = []

            if len(VAL_GUIDES) > 0:
                slope, inter = stats.linregress(train_guide_tas, current_freeAGO_all[:, 0])[:2]
                val_guide_ta = MIRNA_DATA_USE_TPMS.loc[VAL_GUIDES[0]]['guide_TA']
                new_freeago.append(slope * val_guide_ta + inter)


            if options.PASSENGER:
                train_pass_tas = MIRNA_DATA_USE_TPMS.loc[TRAIN_GUIDES]['pass_TA'].values

                if len(VAL_GUIDES) > 0:
                    slope, inter = stats.linregress(train_pass_tas, current_freeAGO_all[:, 1])[:2]
                    val_pass_ta = MIRNA_DATA_USE_TPMS.loc[VAL_GUIDES[0]]['pass_TA']
                    # new_freeago.append(slope * val_pass_ta + inter)
                    new_freeago.append(np.median(current_freeAGO_all[:, 1]))

            if len(VAL_GUIDES) > 0:
                current_freeAGO_all_val = np.concatenate([current_freeAGO_all, np.array([new_freeago])], axis=0).flatten()
            else:
                current_freeAGO_all_val = current_freeAGO_all.flatten()

            # predict repression for validation miRNA
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

            pred_vals_normed = pred_vals - np.mean(pred_vals, axis=1).reshape([-1,1])
            real_vals_normed = real_vals - np.mean(real_vals, axis=1).reshape([-1,1])

            val_losses.append(np.sum(np.square(pred_vals_normed[:, -1] - real_vals_normed[:, -1])))
            current_r2 = stats.linregress(pred_vals_normed[:, -1], real_vals_normed[:, -1])[2]**2
            print(current_r2)
            r2s.append(current_r2)

            if options.LOGDIR is not None:
                eval_names = [
                    'next_kd_batch', 'next_tpm_batch', 'ka_pred', 'pred_repression',
                    'normed_tpms', 'normed_pred_repression', 'loss',
                    'ka_loss', 'repression_loss', 'regularize_term',
                    'freeAGO_mean', 'decay', 'offset_weight', 'utr_ka_values', 'debug', 'learning_rate',
                    'kd_weights', 'merged'
                ]
                evals_dict = {eval_names[ix]: evals[ix] for ix in range(len(eval_names))}

                pretrain_saver.save(sess, os.path.join(PRETRAIN_SAVE_PATH, 'model'), global_step=current_epoch)
                saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=current_epoch)

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
                write_str = 'Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(current_epoch, evals_dict['loss'], evals_dict['ka_loss'], evals_dict['repression_loss'], evals_dict['regularize_term'])
                logfile.write(write_str.encode("utf-8"))
                print(write_str)
                print(evals_dict['kd_weights'])

                write_str = ','.join(['{:.3f}'.format(x) for x in list(sess.run(_ts7_weights).flatten())]) + '\n'
                logfile.write(write_str.encode("utf-8"))
                print(write_str)
                print(current_freeAGO_all)

                fig = plt.figure(figsize=(7, 7))
                plt.scatter(train_guide_tas, current_freeAGO_all[:, 0])

                if len(VAL_GUIDES) > 0:
                    plt.scatter([val_guide_ta], new_freeago[0], color='red')

                plt.savefig(os.path.join(options.LOGDIR, 'train_ta_guide_freeago.png'))
                plt.close()

                if options.PASSENGER:
                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(train_pass_tas, current_freeAGO_all[:, 1])

                    if len(VAL_GUIDES) > 0:
                        plt.scatter([val_pass_ta], new_freeago[1], color='red')

                    plt.savefig(os.path.join(options.LOGDIR, 'train_ta_pass_freeago.png'))
                    plt.close()

                print('val shape:')
                print(real_vals.shape)

                fig = plt.figure(figsize=(7, 7))
                plt.scatter(real_nsites[:, -1], pred_vals[:, -1])
                plt.savefig(os.path.join(options.LOGDIR, 'val_nsites_scatter.png'))
                plt.close()

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

                fig = plt.figure(figsize=(7, 5))
                plt.plot(val_losses)
                plt.savefig(os.path.join(options.LOGDIR, 'val_losses.png'))
                plt.close()

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
                
                print(sess.run(_pred_ka_values, feed_dict={_phase_train: False, _dropout_rate: 0.0, _combined_x: testcase_image}))
                print(sess.run(_pred_ka_values, feed_dict={_phase_train: USE_BATCH_NORM, _dropout_rate: 0.0, _combined_x: testcase_image}))
                print(sess.run(_pred_ka_values, feed_dict={_phase_train: USE_BATCH_NORM, _dropout_rate: options.DROPOUT_RATE, _combined_x: testcase_image}))

            if options.DRY_RUN:
                print(evals_dict['next_tpm_batch']['images'].shape)
                print(evals_dict['next_tpm_batch']['features'].shape)
                print(evals_dict['next_kd_batch']['images'].shape)
                print(evals_dict['next_kd_batch']['labels'].shape)
                print(testcase_image.shape)
                print(evals_dict['next_tpm_batch']['transcripts'])
                print(evals_dict['next_tpm_batch']['nsites'])
                print(evals_dict['next_tpm_batch']['labels'])
                print(evals_dict['next_tpm_batch']['batches'])
                print(evals_dict['utr_ka_values'][-1])
                print(np.max(evals_dict['next_tpm_batch']['images'][-1, :, :], axis=0).reshape([-1, 4]))
                print(np.max(evals_dict['next_tpm_batch']['images'][-1, :, :], axis=1).reshape([-1, 4]))                
                break

        if options.LOGDIR is not None:

            # write final freeAGO concentrations
            final_freeagos = pd.DataFrame(current_freeAGO_all, index=TRAIN_GUIDES, columns = ['guide','pass'])
            final_freeagos['TA'] = train_guide_tas
            final_freeagos.to_csv(os.path.join(options.LOGDIR, 'freeagos.txt'), sep='\t', float_format='%.3f')

        print(np.mean(r2s[-5:]))
        print(current_freeAGO_all_val)
        logfile.close()



