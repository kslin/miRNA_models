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
    parser.add_option("--repression_batch_size", dest="REPRESSION_BATCH_SIZE", type=int)
    parser.add_option("--kd_batch_size", dest="KD_BATCH_SIZE", type=int)
    parser.add_option("--num_epochs", dest="NUM_EPOCHS", type=int)
    parser.add_option("--val_mir", dest="VAL_MIR", help="testing miRNA")
    # parser.add_option("--baseline", dest="BASELINE_METHOD", help="which baseline to use")
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

    if (not os.path.isdir(options.LOGDIR)):
        os.makedirs(options.LOGDIR)

    # SEQLEN must be 12
    SEQLEN = 12

    ### READ miRNA DATA ###
    MIRNA_DATA = pd.read_csv(options.MIR_SEQS, sep='\t', index_col='mir')
    MIRNA_DATA_WITH_RBNS = MIRNA_DATA[MIRNA_DATA['has_rbns']]
    MIRNA_DATA_USE_TPMS = MIRNA_DATA[MIRNA_DATA['use_tpms']]

    ALL_GUIDES = list(MIRNA_DATA_USE_TPMS.index)

    # split miRNAs into training and testing
    if options.VAL_MIR == 'none':
        TRAIN_GUIDES = ALL_GUIDES
        VAL_GUIDES = ['mir139']
        COLORS = ['black'] * len(ALL_GUIDES)
        VAL_IX = ALL_GUIDES.index('mir139')
    else:
        if options.VAL_MIR not in ALL_GUIDES:
            raise ValueError('Test miRNA not in mirseqs file.')
        TRAIN_GUIDES = [m for m in ALL_GUIDES if m != options.VAL_MIR]
        VAL_GUIDES = [options.VAL_MIR]
        COLORS = ['red' if x == options.VAL_MIR else 'black' for x in ALL_GUIDES]
        VAL_IX = ALL_GUIDES.index(options.VAL_MIR)

    if options.VAL_MIR in list(MIRNA_DATA_WITH_RBNS.index):
        TRAIN_MIRS_KDS = [x for x in list(MIRNA_DATA_WITH_RBNS.index) if x != options.VAL_MIR]
        VAL_MIRS_KDS = [options.VAL_MIR]
    else:
        TRAIN_MIRS_KDS = list(MIRNA_DATA_WITH_RBNS.index)
        VAL_MIRS_KDS = []

    NUM_TRAIN_GUIDES = len(TRAIN_GUIDES)

    if options.PASSENGER:
        TRAIN_MIRS = np.array(list(zip(TRAIN_GUIDES, [x + '*' for x in TRAIN_GUIDES]))).flatten().tolist()
        VAL_MIRS = np.array(list(zip(VAL_GUIDES, [x + '*' for x in VAL_GUIDES]))).flatten().tolist()
        ALL_MIRS = np.array(list(zip(ALL_GUIDES, [x + '*' for x in ALL_GUIDES]))).flatten().tolist()
    else:
        TRAIN_MIRS = TRAIN_GUIDES
        ALL_MIRS = ALL_GUIDES

    print("Repression datasets for training: {}".format(TRAIN_GUIDES))
    print("Repression miRNAs for training: {}".format(TRAIN_MIRS))
    print("RBNS miRNAs for training: {}".format(TRAIN_MIRS_KDS))

    # TPM data reader
    tpm_dataset = tf.data.TFRecordDataset(options.TPM_TFRECORDS)
    # tpm_dataset = tpm_dataset.shuffle(buffer_size=1000)

    def _parse_fn_train(x):
        return parse_data_utils._parse_repression_function(x, TRAIN_MIRS, ALL_MIRS, options.MIRLEN, SEQLEN, options.NUM_FEATS)

    def _parse_fn_val(x):
        return parse_data_utils._parse_repression_function(x, ALL_MIRS, ALL_MIRS, options.MIRLEN, SEQLEN, options.NUM_FEATS)

    # preprocess data
    tpm_train_dataset = tpm_dataset.skip(400).shuffle(buffer_size=1000)
    # tpm_train_dataset = tpm_train_dataset.prefetch(options.REPRESSION_BATCH_SIZE)
    tpm_train_dataset = tpm_train_dataset.map(_parse_fn_train, num_parallel_calls=16)
    tpm_val_dataset = tpm_dataset.take(400)
    # tpm_val_dataset = tpm_val_dataset.prefetch(options.REPRESSION_BATCH_SIZE)
    tpm_val_dataset = tpm_val_dataset.map(_parse_fn_val, num_parallel_calls=16)

    # make feedable iterators
    tpm_train_iterator = tpm_train_dataset.make_initializable_iterator()
    tpm_val_iterator = tpm_val_dataset.make_initializable_iterator()

    # create handle for switching between training and validation
    tpm_handle = tf.placeholder(tf.string, shape=[])
    tpm_iterator = tf.data.Iterator.from_string_handle(tpm_handle, tpm_train_dataset.output_types)
    next_tpm_batch = parse_data_utils._build_tpm_batch(tpm_iterator, options.REPRESSION_BATCH_SIZE)

    # KD data reader
    if options.PRETRAIN:
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

    # kd_train_dataset = kd_train_dataset.prefetch(options.KD_BATCH_SIZE)
    # kd_val_dataset = kd_val_dataset.prefetch(1000)

    # shuffle, batch, and map datasets
    kd_train_dataset = kd_train_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
    # kd_train_dataset = kd_train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=None)
    # kd_train_dataset = kd_train_dataset.repeat()  # repeat as long as needed for tpm epochs
    kd_train_dataset = kd_train_dataset.map(parse_data_utils._parse_log_kd_function, num_parallel_calls=16)

    # re-balance KD data towards high-affinity sites
    kd_train_dataset = kd_train_dataset.filter(parse_data_utils._filter_kds)
    kd_train_dataset = kd_train_dataset.batch(options.KD_BATCH_SIZE, drop_remainder=True)

    kd_val_dataset = kd_val_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=None)
    kd_val_dataset = kd_val_dataset.apply(tf.data.experimental.map_and_batch(
            map_func=parse_data_utils._parse_log_kd_function, batch_size=1000))
    # kd_val_dataset = kd_val_dataset.map(parse_data_utils._parse_log_kd_function)
    # kd_val_dataset = kd_val_dataset.batch(1000)

    # make feedable iterators
    kd_train_iterator = kd_train_dataset.make_initializable_iterator()
    kd_val_iterator = kd_val_dataset.make_initializable_iterator()

    # create handle for switching between training and validation
    kd_handle = tf.placeholder(tf.string, shape=[])
    kd_iterator = tf.data.Iterator.from_string_handle(kd_handle, kd_train_dataset.output_types)
    next_kd_batch_mirs, next_kd_batch_images, next_kd_batch_labels = kd_iterator.get_next()

    # add random sequences generator
    def gen():
        while True:
            random_mirseq = utils.generate_random_seq(options.MIRLEN)
            random_target = utils.get_target_no_match(random_mirseq, SEQLEN)
            random_image = np.outer(utils.one_hot_encode(random_mirseq), utils.one_hot_encode(random_target))

            rbns_mir = np.random.choice(TRAIN_MIRS_KDS)
            rbns_mirseq = MIRNA_DATA.loc[rbns_mir]['guide_seq'][:options.MIRLEN]
            rbns_target = utils.get_target_no_match(rbns_mirseq, SEQLEN)
            rbns_image = np.outer(utils.one_hot_encode(rbns_mirseq), utils.one_hot_encode(rbns_target))
            yield np.array([b'random', rbns_mir.encode('utf-8')]), np.stack([random_image, rbns_image]), np.array([[0.0], [0.0]])

    random_seq_dataset = tf.data.Dataset.from_generator(gen, (tf.string, tf.float32, tf.float32))
    random_seq_iterator = random_seq_dataset.make_initializable_iterator()
    random_seq_mirs, random_seq_images, random_seq_labels = random_seq_iterator.get_next()

    next_kd_batch = {
        'mirs': tf.concat([next_kd_batch_mirs, random_seq_mirs], axis=0),
        'images': tf.concat([next_kd_batch_images, random_seq_images], axis=0),
        'labels': tf.nn.relu(-1 * tf.concat([next_kd_batch_labels, random_seq_labels], axis=0))
    }

    # create placeholders for input data
    _keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    _phase_train = tf.placeholder(tf.bool, name='phase_train')

    # build KA predictor
    _combined_x = tf.concat([next_kd_batch['images'], next_tpm_batch['images']], axis=0)
    _combined_x_4D = tf.expand_dims((_combined_x * 4.0) - 0.25, axis=3)  # reshape, zero-center input
    _pred_ka_values, _cnn_weights = model.seq2ka_predictor(
        _combined_x_4D, _keep_prob, _phase_train,
        options.HIDDEN1, options.HIDDEN2, options.HIDDEN3, options.MIRLEN, SEQLEN
    )  # pred ka

    # split data into biochem and repression and get biochem loss
    _pred_biochem = _pred_ka_values[:tf.shape(next_kd_batch['images'])[0], :]
    # _ka_loss = (tf.nn.l2_loss(tf.subtract(tf.nn.relu(_pred_biochem), next_kd_batch['labels'])))# / options.KD_BATCH_SIZE
    _ka_loss_weights = tf.sqrt(next_kd_batch['labels'] + 1.0)
    _ka_loss = tf.nn.l2_loss(tf.subtract(tf.nn.relu(_pred_biochem), next_kd_batch['labels']) * _ka_loss_weights)
    _utr_ka_values = _pred_ka_values[tf.shape(next_kd_batch['images'])[0]:, :]

    # make model saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=options.NUM_EPOCHS)

    # create freeAGO concentration variables
    _freeAGO_mean = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(-5.0))
    _freeAGO_offset = tf.get_variable('freeAGO_offset', shape=[len(TRAIN_MIRS)], initializer=tf.constant_initializer(0.0))
    _freeAGO_all = _freeAGO_mean + _freeAGO_offset + tf.tile(tf.constant([0.0, -1.0]), tf.constant([NUM_TRAIN_GUIDES]))
    
    if options.VAL_MIR  == 'none':
        _freeAGO_all_val = _freeAGO_all
    else:
        if options.PASSENGER:
            _freeAGO_all_val = tf.concat([
                _freeAGO_all[:(VAL_IX * 2)],
                tf.reshape(tf.reduce_mean(tf.reshape(_freeAGO_all, [-1, 2]), axis=0), [2]),
                _freeAGO_all[(VAL_IX * 2):]], axis=0)
        else:
            _freeAGO_all_val = tf.concat([
                _freeAGO_all[:VAL_IX],
                tf.reshape(tf.reduce_mean(_freeAGO_all, axis=0), [1]),
                _freeAGO_all[VAL_IX:]], axis=0)

    # make feature weights
    weights_init = np.array([0.001]*options.NUM_FEATS).reshape([options.NUM_FEATS, 1])

    # create ts7 weight variable
    with tf.name_scope('ts7_layer'):
        with tf.name_scope('weights'):
            _ts7_weights = tf.get_variable("ts7_weights", shape=[options.NUM_FEATS, 1],
                                        initializer=tf.constant_initializer(weights_init))
            tf.add_to_collection('weight', _ts7_weights)
            _decay = tf.get_variable('decay', initializer=-1.0)
            _ts7_bias = tf.get_variable('ts7_bias', initializer=1.0)
            # _ts7_weights = tf.concat([tf.constant(np.array([[1.0]]), dtype=tf.float32), _ts7_weights], axis=0)

    # get logfc prediction
    _pred_logfc, _pred_logfc_normed, _repression_y_normed = model.get_pred_logfc_separate(
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

    _pred_logfc_val, _pred_logfc_val_normed, _repression_y_val_normed = model.get_pred_logfc_separate(
        _utr_ka_values,
        _freeAGO_all_val,
        next_tpm_batch,
        _ts7_weights,
        _ts7_bias,
        _decay,
        options.REPRESSION_BATCH_SIZE,
        options.PASSENGER,
        len(ALL_GUIDES),
        'pred_logfc_val',
        options.LOSS_TYPE
    )

    print('pred_logfc: {}'.format(_pred_logfc))
    print(_pred_logfc_normed)
    print(_repression_y_normed)
    _repression_loss = 5.0 * tf.nn.l2_loss(tf.subtract(_pred_logfc_normed, _repression_y_normed))# / (options.REPRESSION_BATCH_SIZE * NUM_TRAIN_GUIDES)

    if options.PASSENGER:
        offset_weight = tf.reduce_sum(tf.reshape(_freeAGO_all, [-1, 2])[:, 0])
    else:
        offset_weight = tf.reduce_sum(_freeAGO_all)

    # define regularizer
    _weight_regularize = tf.multiply(
        offset_weight +
        tf.nn.l2_loss(_cnn_weights['w1']) +
        tf.nn.l2_loss(_cnn_weights['w2']) +
        tf.nn.l2_loss(_cnn_weights['w3']) +
        tf.nn.l2_loss(_cnn_weights['w4']),
        options.LAMBDA
    )

    # define loss and train_step
    if options.PRETRAIN:
        _loss = _ka_loss
        
    else:
        _loss = _ka_loss + _repression_loss + _weight_regularize


    _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(_update_ops):
        _train_step = tf.train.AdamOptimizer(options.LEARNING_RATE).minimize(_loss)

    if not options.DRY_RUN:
        logfile = open(os.path.join(options.LOGDIR, 'out.log'), 'w', -1)

    # train model
    losses = []
    means = []
    r2s = []
    conf = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=24) 
    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())

        if options.LOAD_MODEL is not None:
            latest = tf.train.latest_checkpoint(options.LOAD_MODEL)
            print('Restoring from {}'.format(latest))
            saver.restore(sess, latest)

        for current_epoch in range(options.NUM_EPOCHS):
            sess.run(tpm_train_iterator.initializer)
            sess.run(kd_train_iterator.initializer)
            sess.run(random_seq_iterator.initializer)

            # get training and validation handles
            tpm_train_handle = sess.run(tpm_train_iterator.string_handle())
            kd_train_handle = sess.run(kd_train_iterator.string_handle())

            train_feed_dict = {_phase_train: True, _keep_prob: 0.5, tpm_handle: tpm_train_handle, kd_handle: kd_train_handle}

            if options.DRY_RUN:
                sess.run(tpm_val_iterator.initializer)
                sess.run(kd_val_iterator.initializer)
                tpm_val_handle = sess.run(tpm_val_iterator.string_handle())
                kd_val_handle = sess.run(kd_val_iterator.string_handle())
                val_feed_dict = {_phase_train: False, _keep_prob: 1.0, tpm_handle: tpm_val_handle, kd_handle: kd_val_handle}

                print(COLORS)

                train_evals = sess.run([
                    next_kd_batch,  #0
                    _pred_biochem,  #1
                    next_tpm_batch,  #2
                    _repression_y_normed,  #3
                    _pred_logfc_normed,  #4
                    _train_step,  #5
                    random_seq_mirs,
                    random_seq_images,
                    random_seq_labels,
                    _utr_ka_values
                ], feed_dict=train_feed_dict)

                print(train_evals[0]['mirs'])
                print(train_evals[0]['labels'].shape, train_evals[1].shape)
                print(train_evals[0]['labels'], train_evals[1])
                print(train_evals[2]['images'].shape)
                print(train_evals[2]['labels'].shape)
                print(train_evals[2]['nsites'].shape, np.sum(train_evals[2]['nsites']))
                print(train_evals[3].shape, train_evals[4].shape)
                print(train_evals[6])
                print(train_evals[7].shape, train_evals[8])
                print(np.min(train_evals[-1]), np.max(train_evals[-1]))

                fig = plt.figure(figsize=(7, 7))
                plt.scatter(train_evals[1].flatten(), train_evals[0]['labels'].flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'train_ka_scatter.png'))
                plt.close()

                val_evals = sess.run([
                    next_kd_batch,
                    _pred_biochem,
                    next_tpm_batch,
                    _repression_y_val_normed,
                    _pred_logfc_val_normed
                ], feed_dict=val_feed_dict)

                print(list(set(list(val_evals[0]['mirs']))))
                print(val_evals[0]['labels'].shape, val_evals[1].shape)
                print(val_evals[2]['images'].shape)
                print(val_evals[2]['labels'].shape)
                print(val_evals[2]['nsites'].shape, np.sum(val_evals[2]['nsites']))
                print(val_evals[3].shape, val_evals[4].shape)

                fig = plt.figure(figsize=(7, 7))
                plt.scatter(val_evals[1].flatten(), val_evals[0]['labels'].flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'val_ka_scatter.png'))
                plt.close()

                sess.run(tpm_val_iterator.initializer)
                pred_vals, real_vals = [], []
                while True:
                    try:
                        temp_tpm_batch = sess.run(next_tpm_batch, feed_dict={tpm_handle: tpm_val_handle})
                        real_vals.append(temp_tpm_batch['labels'])
                        ka_vals = sess.run(_pred_ka_values, feed_dict={_phase_train: False, _keep_prob: 1.0, _combined_x: temp_tpm_batch['images']})
                        pred_vals.append(sess.run(_pred_logfc_val,
                            feed_dict={
                                _utr_ka_values: ka_vals,
                                next_tpm_batch['nsites']: temp_tpm_batch['nsites'],
                                next_tpm_batch['features']: temp_tpm_batch['features'],
                                next_tpm_batch['labels']: temp_tpm_batch['labels'],
                            }))
                    except tf.errors.OutOfRangeError:
                        break

                pred_vals = np.concatenate(pred_vals)
                pred_vals = pred_vals[:, VAL_IX] - np.mean(pred_vals, axis=1)

                real_vals = np.concatenate(real_vals)
                real_vals = real_vals[:, VAL_IX] - np.mean(real_vals, axis=1)
                
                print(pred_vals.shape, real_vals.shape)
                print(np.mean(pred_vals), np.mean(real_vals))

                sess.run(_train_step, feed_dict=train_feed_dict)

                sess.run(tpm_val_iterator.initializer)
                pred_vals, real_vals = [], []
                while True:
                    try:
                        temp_tpm_batch = sess.run(next_tpm_batch, feed_dict={tpm_handle: tpm_val_handle})
                        real_vals.append(temp_tpm_batch['labels'])
                        ka_vals = sess.run(_pred_ka_values, feed_dict={_phase_train: False, _keep_prob: 1.0, _combined_x: temp_tpm_batch['images']})
                        pred_vals.append(sess.run(_pred_logfc_val,
                            feed_dict={
                                _utr_ka_values: ka_vals,
                                next_tpm_batch['nsites']: temp_tpm_batch['nsites'],
                                next_tpm_batch['features']: temp_tpm_batch['features'],
                                next_tpm_batch['labels']: temp_tpm_batch['labels'],
                            }))
                    except tf.errors.OutOfRangeError:
                        break

                pred_vals = np.concatenate(pred_vals)
                pred_vals = pred_vals[:, VAL_IX] - np.mean(pred_vals, axis=1)

                real_vals = np.concatenate(real_vals)
                real_vals = real_vals[:, VAL_IX] - np.mean(real_vals, axis=1)

                print(pred_vals.shape, real_vals.shape)
                print(np.mean(pred_vals), np.mean(real_vals))

                break

            time_start = time.time()
            while True:
                try:
                    evals = sess.run([
                        next_kd_batch,
                        _pred_biochem,
                        _repression_y_normed,
                        _pred_logfc_normed,
                        _train_step,
                        _loss,
                        _ka_loss,
                        _repression_loss,
                        _weight_regularize,
                        _freeAGO_mean
                    ], feed_dict=train_feed_dict)

                except tf.errors.OutOfRangeError:

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

                    colors = ['black'] * options.KD_BATCH_SIZE + ['red'] * 2
                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(evals[1].flatten(), evals[0]['labels'].flatten(), color=colors)
                    plt.savefig(os.path.join(options.LOGDIR, 'train_ka_scatter.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(evals[3].flatten(), evals[2].flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'train_repression_scatter.png'))
                    plt.close()

                    losses.append(evals[5])
                    fig = plt.figure(figsize=(7, 5))
                    plt.plot(losses)
                    plt.savefig(os.path.join(options.LOGDIR, 'train_losses.png'))
                    plt.close()

                    means.append(evals[9])
                    fig = plt.figure(figsize=(7, 5))
                    plt.plot(means)
                    plt.savefig(os.path.join(options.LOGDIR, 'train_freeAGO_means.png'))
                    plt.close()

                    # logfile.write('Time for epoch: {}\n'.format(time.time() - time_start))
                    logfile.write('Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(current_epoch, evals[5], evals[6], evals[7], evals[8]))
                    print('Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(current_epoch, evals[5], evals[6], evals[7], evals[8]))
                    print(sess.run(_ts7_weights).flatten())

                    sess.run(tpm_val_iterator.initializer)
                    tpm_val_handle = sess.run(tpm_val_iterator.string_handle())
                    pred_vals, real_vals = [], []
                    while True:
                        try:
                            temp_tpm_batch = sess.run(next_tpm_batch, feed_dict={tpm_handle: tpm_val_handle})
                            real_vals.append(temp_tpm_batch['labels'])
                            ka_vals = sess.run(_pred_ka_values, feed_dict={_phase_train: False, _keep_prob: 1.0, _combined_x: temp_tpm_batch['images']})
                            pred_vals.append(sess.run(_pred_logfc_val,
                                feed_dict={
                                    _utr_ka_values: ka_vals,
                                    next_tpm_batch['nsites']: temp_tpm_batch['nsites'],
                                    next_tpm_batch['features']: temp_tpm_batch['features'],
                                    next_tpm_batch['labels']: temp_tpm_batch['labels'],
                                }))
                        except tf.errors.OutOfRangeError:
                            break

                    pred_vals = np.concatenate(pred_vals)
                    pred_vals = pred_vals[:, VAL_IX] - np.mean(pred_vals, axis=1)

                    real_vals = np.concatenate(real_vals)
                    real_vals = real_vals[:, VAL_IX] - np.mean(real_vals, axis=1)

                    r2s.append(stats.linregress(pred_vals, real_vals)[2]**2)
                    fig = plt.figure(figsize=(7, 5))
                    plt.plot(r2s)
                    plt.savefig(os.path.join(options.LOGDIR, 'val_r2s.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(pred_vals, real_vals)
                    plt.savefig(os.path.join(options.LOGDIR, 'val_tpm_scatter.png'))
                    plt.close()

                    sess.run(kd_val_iterator.initializer)
                    kd_val_handle = sess.run(kd_val_iterator.string_handle())
                    temp_kd_batch = sess.run(next_kd_batch, feed_dict={kd_handle: kd_val_handle})
                    ka_vals = sess.run(_pred_ka_values, feed_dict={_phase_train: False, _keep_prob: 1.0, _combined_x: temp_kd_batch['images']})

                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(ka_vals.flatten(), temp_kd_batch['labels'].flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'val_ka_scatter.png'))
                    plt.close()

                    break

    if not options.DRY_RUN:
        logfile.close()

