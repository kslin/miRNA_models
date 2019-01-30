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

import tf_helpers
import parse_data_utils

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


def get_pred_logfc(_utr_ka_values, _freeAGO_all, _tpm_batch, _ts7_weights, batch_size, passenger, num_guides, name, loss_type):
    if passenger:
        nsites_mir = tf.reduce_sum(tf.reshape(_tpm_batch['nsites'], [num_guides * batch_size, 2]), axis=1)
        num_mirs = num_guides * 2
    else:
        nsites_mir = tf.reshape(_tpm_batch['nsites'], [num_guides * batch_size])
        num_mirs = num_guides

    _freeAGO_tiled = tf.tile(_freeAGO_all, tf.constant([batch_size]))
    _freeAGO_tiled = tf.concat([tf.tile(_freeAGO_tiled[ix: ix + 1], _tpm_batch['nsites'][ix: ix + 1]) for ix in range(num_mirs * batch_size)], axis=0)

    _nbound = tf.sigmoid(_utr_ka_values + tf.expand_dims(_freeAGO_tiled, axis=1))
    _all_feats = tf.concat([_nbound, _tpm_batch['features']], axis=1)
    _pred_logfc_ind_sites = -1 * (tf.squeeze(tf.matmul(_all_feats, _ts7_weights)))
    _pred_logfc_splits = tf.split(_pred_logfc_ind_sites, nsites_mir)
    _pred_logfc = tf.reshape(tf.stack([tf.reduce_sum(x) for x in _pred_logfc_splits]), [batch_size, num_guides], name=name)

    if loss_type == 'MEAN_CENTER':
        _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1, 1])
        _repression_y_normed = _tpm_batch['labels'] - tf.reshape(tf.reduce_mean(_tpm_batch['labels'], axis=1), [-1, 1])
    else:
        _pred_logfc_normed = _pred_logfc
        _repression_y_normed = _tpm_batch['labels']

    return _freeAGO_tiled, _pred_logfc, _pred_logfc_normed, _repression_y_normed


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_tfrecords", dest="TPM_TFRECORDS", help="tpm data in tfrecord format")
    parser.add_option("--kd_tfrecords", dest="KD_TFRECORDS", help="kd data in tfrecord format")
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--num_feats", dest="NUM_FEATS", type=int)
    parser.add_option("--repression_batch_size", dest="REPRESSION_BATCH_SIZE", type=int)
    parser.add_option("--kd_batch_size", dest="KD_BATCH_SIZE", type=int)
    parser.add_option("--num_epochs", dest="NUM_EPOCHS", type=int)
    parser.add_option("--val_mir", dest="VAL_MIR", help="testing miRNA")
    parser.add_option("--baseline", dest="BASELINE_METHOD", help="which baseline to use")
    parser.add_option("--loss_type", dest="LOSS_TYPE", help="which loss strategy")
    parser.add_option("--lambda", dest="LAMBDA", help="regularizer weight", type=float)
    parser.add_option("--lr", dest="LEARNING_RATE", help="starting learning rate", type=float)
    parser.add_option("--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("--load_model", dest="LOAD_MODEL", help="if supplied, load latest model from this directory", default=None)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--dry_run", dest="DRY_RUN", help="if true, do dry run", default=False, action='store_true')

    (options, args) = parser.parse_args()

    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if (not os.path.isdir(options.LOGDIR)) and (not options.DRY_RUN):
        os.makedirs(options.LOGDIR)

    # SEQLEN must be 12
    SEQLEN = 12

    ### READ miRNA DATA ###
    MIRNA_DATA = pd.read_csv(options.MIR_SEQS, sep='\t')
    MIRNA_DATA_WITH_RBNS = MIRNA_DATA[MIRNA_DATA['has_rbns']]
    MIRNA_DATA_USE_TPMS = MIRNA_DATA[MIRNA_DATA['use_tpms']]

    ALL_GUIDES = list(MIRNA_DATA_USE_TPMS['mir'].values)

    # split miRNAs into training and testing
    if options.VAL_MIR == 'none':
        TRAIN_GUIDES = ALL_GUIDES
        VAL_GUIDES = ['mir139']
        COLORS = ['black'] * len(ALL_GUIDES)
        VAL_IX = 0
    else:
        if options.VAL_MIR not in ALL_GUIDES:
            raise ValueError('Test miRNA not in mirseqs file.')
        TRAIN_GUIDES = [m for m in ALL_GUIDES if m != options.VAL_MIR]
        VAL_GUIDES = [options.VAL_MIR]
        COLORS = ['red' if x == options.VAL_MIR else 'black' for x in ALL_GUIDES]
        VAL_IX = ALL_GUIDES.index(options.VAL_MIR)

    if options.VAL_MIR in MIRNA_DATA_WITH_RBNS['mir'].values:
        TRAIN_MIRS_KDS = [x for x in MIRNA_DATA_WITH_RBNS['mir'].values if x != options.VAL_MIR]
        VAL_MIRS_KDS = [options.VAL_MIR]
    else:
        TRAIN_MIRS_KDS = MIRNA_DATA_WITH_RBNS['mir'].values
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

    # create freeAGO concentration variables
    _freeAGO_mean = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(-4.0))
    _freeAGO_offset = tf.get_variable('freeAGO_offset', shape=[len(ALL_MIRS)], initializer=tf.constant_initializer(0.0))
    _freeAGO_offset
    # _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset', shape=[NUM_TRAIN_GUIDES, 1],
    #     initializer=tf.constant_initializer(0.0))

    if options.PASSENGER:
        _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset', shape=[NUM_TRAIN_GUIDES, 1],
            initializer=tf.constant_initializer(-1.0))
        _freeAGO_all_stacked = tf.concat([_freeAGO_guide_offset + _freeAGO_mean, _freeAGO_pass_offset + _freeAGO_mean], axis=1)
        _freeAGO_all = tf.reshape(_freeAGO_all_stacked, [NUM_TRAIN_GUIDES * 2], name='freeAGO_all')
        if options.VAL_MIR  == 'none':
            _freeAGO_all_val = _freeAGO_all
        else:
            _freeAGO_all_val = tf.concat([
                _freeAGO_all[:(VAL_IX * 2)],
                tf.reshape(tf.reduce_mean(_freeAGO_all_stacked, axis=0), [2]),
                _freeAGO_all[(VAL_IX * 2):]], axis=0)
    else:
        _freeAGO_all = tf.squeeze(_freeAGO_guide_offset + _freeAGO_mean, name='freeAGO_all')
        if options.VAL_MIR  == 'none':
            _freeAGO_all_val = _freeAGO_all
        else:
            _freeAGO_all_val = tf.concat([
                _freeAGO_all[:VAL_IX],
                tf.reshape(tf.reduce_mean(_freeAGO_all_stacked, axis=0), [2]),
                _freeAGO_all[VAL_IX:]], axis=0)

    # _freeAGO_mean = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(-4.0))
    # _freeAGO_offset = tf.get_variable('freeAGO_offset', shape=(NUM_TRAIN_GUIDES * 2),
    #     initializer=tf.constant_initializer(0.0))
    # _freeAGO_all = _freeAGO_mean + _freeAGO_offset


    ######

    # batch_size=2
    # _freeAGO_mean = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(-4.0))
    # _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset', shape=[2, 1],
    #         initializer=tf.constant_initializer(0.0))
    # _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset', shape=[2, 1],
    #             initializer=tf.constant_initializer(-1.0))
    # _freeAGO_all_stacked = tf.concat([_freeAGO_guide_offset + _freeAGO_mean, _freeAGO_pass_offset + _freeAGO_mean], axis=1)
    # _freeAGO_all = tf.reshape(_freeAGO_all_stacked, [NUM_TRAIN_GUIDES * 2], name='freeAGO_all')

    tpm_dataset = tf.data.TFRecordDataset(options.TPM_TFRECORDS)
    tpm_train_dataset = tpm_dataset.shuffle(buffer_size=1000)

    def _parse_fn_train(x):
        return parse_data_utils._parse_repression_function(x, TRAIN_MIRS, ALL_MIRS, options.MIRLEN, SEQLEN, options.NUM_FEATS)

    # def _parse_fn_val(x):
    #     return parse_data_utils._parse_repression_function(x, ALL_MIRS, ALL_MIRS, options.MIRLEN, SEQLEN, options.NUM_FEATS)

    # preprocess data
    tpm_train_dataset = tpm_train_dataset.map(_parse_fn_train)
    # tpm_train_dataset = tpm_train_dataset.batch(batch_size=options.REPRESSION_BATCH_SIZE)
    # # tpm_val_dataset = tpm_dataset.map(_parse_fn_val)

    # # make feedable iterators
    tpm_train_iterator = tpm_train_dataset.make_initializable_iterator()
    # # tpm_val_iterator = tpm_val_dataset.make_initializable_iterator()

    # # create handle for switching between training and validation
    # # tpm_handle = tf.placeholder(tf.string, shape=[])
    # # tpm_iterator = tf.data.Iterator.from_string_handle(tpm_handle, tpm_train_dataset.output_types)
    next_tpm_batch = parse_data_utils._build_tpm_batch(tpm_train_iterator, options.REPRESSION_BATCH_SIZE, options.PASSENGER)
    # var = tf.get_variable('var', shape=(), initializer=tf.constant_initializer(-4))
    # Y = next_tpm_batch['images'] + var

    total_num = NUM_TRAIN_GUIDES * 2 * options.REPRESSION_BATCH_SIZE
    _freeAGO_tiled = tf.reshape(tf.tile(_freeAGO_all, tf.constant([options.REPRESSION_BATCH_SIZE])), [total_num, 1])
    # _freeAGO_tiled = tf.concat([tf.tile(_freeAGO_tiled[ix: ix + 1], next_tpm_batch['nsites'][ix: ix + 1]) for ix in range(NUM_TRAIN_GUIDES * 2 * options.REPRESSION_BATCH_SIZE)], axis=0)

    # _freeAGO_tiled = []
    # ix = 0
    # for _ in range(options.REPRESSION_BATCH_SIZE):
    #     for iy in range(NUM_TRAIN_GUIDES * 2):
    #         _freeAGO_tiled.append(tf.tile(_freeAGO_all[iy: iy + 1], next_tpm_batch['nsites'][ix: ix + 1]))
    #         ix += 1
    # _freeAGO_tiled = tf.concat(_freeAGO_tiled, axis=0)

    # _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(_update_ops):
    #     _train_step = tf.train.AdamOptimizer(options.LEARNING_RATE).minimize(_freeAGO_all)
        
    
    # freeA_tiled = tf.tile(_freeAGO_all, tf.constant([options.REPRESSION_BATCH_SIZE], dtype=tf.int32))
    # nsites = tf.constant(np.random.randint(3, size=options.REPRESSION_BATCH_SIZE * NUM_TRAIN_GUIDES * 2), dtype=tf.float32)
    # _freeAGO_tiled = tf.constant(np.random.randint(3, size=options.REPRESSION_BATCH_SIZE * NUM_TRAIN_GUIDES * 2), dtype=tf.float32)
    nsites = tf.reshape(next_tpm_batch['nsites'], [total_num, 1])
    # nsites2 = tf.cast(nsites, tf.float32) + _freeAGO_tiled
    # nsites2 = tf.concat([tf.tile(_freeAGO_tiled[0:1], nsites[0:1]), tf.tile(_freeAGO_tiled[1:2], nsites[1:2])], axis=0)

    print(_freeAGO_tiled, nsites)
    elems = (_freeAGO_tiled, tf.cast(nsites, tf.float32))
    # nsites2 = tf.map_fn(lambda x: x[0] + x[1], elems, dtype=tf.float32)
    # nsites2 = tf.map_fn(lambda x: tf.tile(x[0], x[1]), elems, dtype=tf.float32)
    def loop_body(i, ta):
        return i + 1, ta.write(i, tf.random_normal((lengths[i],), 0, 1))
    nsites2 = tf.while_loop(lambda i, ta: i < total_num, loop_body, [0, init_array])
    
    # nsites2 = tf.concat([tf.tile(_freeAGO_tiled[ix:ix+1], nsites[ix:ix+1]) for ix in range(60)], axis=0)
    print(total_num, nsites, nsites2)
    # print(nsites1)
    # print(nsites2)
    # X = [tf.tile(freeA_tiled[ix:ix+1], nsites2[ix:ix+1]) for ix in range(options.REPRESSION_BATCH_SIZE * NUM_TRAIN_GUIDES * 2)]
    # Y = tf.concat(X, axis=0)
    # grad = tf.gradients(ys=Y, xs=_freeAGO_mean)
    print(tf.trainable_variables())
    _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(_update_ops):
        _train_step = tf.train.AdamOptimizer(0.001).minimize(tf.cast(nsites2, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tpm_train_iterator.initializer)
        print(sess.run(nsites2))
        # blah = sess.run([nsites, _freeAGO_tiled, _freeAGO_all, _train_step])
        # print(blah[0].shape)
        # print(blah[1].shape)
        # print(blah[2].shape)
        # print(blah[0][:16])
        # print(blah[1][:16])
    #     print(sess.run(nsites1).shape)
    #     print(sess.run(nsites2).shape)

    sys.exit()


    ######

    # TPM data reader
    tpm_dataset = tf.data.TFRecordDataset(options.TPM_TFRECORDS)
    tpm_train_dataset = tpm_dataset.shuffle(buffer_size=1000)

    def _parse_fn_train(x):
        return parse_data_utils._parse_repression_function(x, TRAIN_MIRS, ALL_MIRS, options.MIRLEN, SEQLEN, options.NUM_FEATS)

    def _parse_fn_val(x):
        return parse_data_utils._parse_repression_function(x, ALL_MIRS, ALL_MIRS, options.MIRLEN, SEQLEN, options.NUM_FEATS)

    # preprocess data
    tpm_train_dataset = tpm_train_dataset.map(_parse_fn_train)
    tpm_val_dataset = tpm_dataset.map(_parse_fn_val)

    # make feedable iterators
    tpm_train_iterator = tpm_train_dataset.make_initializable_iterator()
    tpm_val_iterator = tpm_val_dataset.make_initializable_iterator()

    # create handle for switching between training and validation
    tpm_handle = tf.placeholder(tf.string, shape=[])
    tpm_iterator = tf.data.Iterator.from_string_handle(tpm_handle, tpm_train_dataset.output_types)
    next_tpm_batch = parse_data_utils._build_tpm_batch(tpm_iterator, options.REPRESSION_BATCH_SIZE, options.PASSENGER)

    # KD data reader
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

    # shuffle, batch, and map datasets
    kd_train_dataset = kd_train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=None)
    kd_train_dataset = kd_train_dataset.map(parse_data_utils._parse_log_kd_function)
    kd_train_dataset = kd_train_dataset.batch(options.KD_BATCH_SIZE, drop_remainder=True)

    kd_val_dataset = kd_val_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=None)
    kd_val_dataset = kd_val_dataset.map(parse_data_utils._parse_log_kd_function)
    kd_val_dataset = kd_val_dataset.batch(1000, drop_remainder=True)

    # create handle for switching between training and validation
    kd_handle = tf.placeholder(tf.string, shape=[])
    kd_iterator = tf.data.Iterator.from_string_handle(kd_handle, kd_train_dataset.output_types)
    next_kd_batch_mirs, next_kd_batch_images, next_kd_batch_labels = kd_iterator.get_next()

    # add random sequences generator
    # def gen():
    #     for i in itertools.count(1):
    #         yield (i, [1] * i)
    # random_seq_dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
    # random_seq_images, random_seq_labels = 

    next_ka_batch_labels = tf.nn.relu(-1 * next_kd_batch_labels)

    # make feedable iterators
    kd_train_iterator = kd_train_dataset.make_initializable_iterator()
    kd_val_iterator = kd_val_dataset.make_initializable_iterator()

    # create placeholders for input data
    _keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    _phase_train = tf.placeholder(tf.bool, name='phase_train')

    # create ts7 weight variable
    with tf.name_scope('ts7_layer'):
        with tf.name_scope('weights'):
            _ts7_weights = tf.get_variable("ts7_weights", shape=[options.NUM_FEATS + 1, 1],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.add_to_collection('weight', _ts7_weights)

    # build KA predictor
    _combined_x = tf.concat([next_tpm_batch['images'], next_kd_batch_images], axis=0)
    _combined_x_4D = tf.expand_dims((_combined_x * 4.0) - 0.25, axis=3)  # reshape, zero-center input
    _pred_ka_values, _cnn_weights = tf_helpers.seq2ka_predictor(_combined_x_4D, _keep_prob, _phase_train)  # pred ka

    # split data into biochem and repression and get biochem loss
    if options.KD_BATCH_SIZE == 0:
        _pred_biochem = tf.constant(np.array([[0]]))
        _ka_loss = tf.constant(0.0)
        _utr_ka_values = _pred_ka_values
    else:
        _pred_biochem = _pred_ka_values[-1 * tf.shape(next_kd_batch_images)[0]:, :]
        _ka_loss = (tf.nn.l2_loss(tf.subtract(_pred_biochem, next_ka_batch_labels)))# / options.KD_BATCH_SIZE
        _utr_ka_values = _pred_ka_values[:-1 * tf.shape(next_kd_batch_images)[0], :]

    # get logfc prediction
    _freeAGO_tiled, _pred_logfc, _pred_logfc_normed, _repression_y_normed = get_pred_logfc(
        _utr_ka_values,
        _freeAGO_all,
        next_tpm_batch,
        _ts7_weights,
        options.REPRESSION_BATCH_SIZE,
        options.PASSENGER,
        NUM_TRAIN_GUIDES,
        'pred_logfc_train',
        options.LOSS_TYPE
    )

    _freeAGO_tiled_val, _pred_logfc_val, _pred_logfc_val_normed, _repression_y_val_normed = get_pred_logfc(
        _utr_ka_values,
        _freeAGO_all_val,
        next_tpm_batch,
        _ts7_weights,
        options.REPRESSION_BATCH_SIZE,
        options.PASSENGER,
        len(ALL_GUIDES),
        'pred_logfc_val',
        options.LOSS_TYPE
    )

    print('pred_logfc: {}'.format(_pred_logfc))
    print(_pred_logfc_normed)
    print(_repression_y_normed)
    _repression_loss = tf.nn.l2_loss(tf.subtract(_pred_logfc_normed, _repression_y_normed))# / (options.REPRESSION_BATCH_SIZE * NUM_TRAIN_GUIDES)

    # define regularizer
    _weight_regularize = tf.multiply(
        tf.nn.l2_loss(_freeAGO_guide_offset) +
        tf.nn.l2_loss(_cnn_weights['w1']) +
        tf.nn.l2_loss(_cnn_weights['w2']) +
        tf.nn.l2_loss(_cnn_weights['w3']) +
        tf.nn.l2_loss(_cnn_weights['w4']),
        # tf.nn.l2_loss(_ts7_weights),
        options.LAMBDA
    )

    # define loss and train_step
    _loss = _ka_loss + _repression_loss + _weight_regularize

    # _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(_update_ops):
    #     _train_step = tf.train.AdamOptimizer(options.LEARNING_RATE).minimize(_loss)

    # make model saver
    saver = tf.train.Saver(max_to_keep=options.NUM_EPOCHS + 1)

    logfile = open(os.path.join(options.LOGDIR, 'out.log'), 'w', -1)

    # train model
    losses = []
    means = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for current_epoch in range(options.NUM_EPOCHS):
            sess.run(tpm_train_iterator.initializer)
            sess.run(tpm_val_iterator.initializer)
            sess.run(kd_train_iterator.initializer)
            sess.run(kd_val_iterator.initializer)

            # get training and validation handles
            tpm_train_handle = sess.run(tpm_train_iterator.string_handle())
            tpm_val_handle = sess.run(tpm_val_iterator.string_handle())
            kd_train_handle = sess.run(kd_train_iterator.string_handle())
            kd_val_handle = sess.run(kd_val_iterator.string_handle())

            train_feed_dict = {_phase_train: True, _keep_prob: 0.5, tpm_handle: tpm_train_handle, kd_handle: kd_train_handle}
            val_feed_dict = {_phase_train: False, _keep_prob: 1.0, tpm_handle: tpm_val_handle, kd_handle: kd_val_handle}

            if options.DRY_RUN:
                print(COLORS)
                # print(tf.trainable_variables())

                # blah = sess.run(_grads, feed_dict=val_feed_dict)
                # print(blah)

                # blah = sess.run([next_tpm_batch, _freeAGO_tiled, _utr_ka_values, _pred_logfc], feed_dict=train_feed_dict)
                # print(blah[0]['images'].shape)
                # print(np.sum(blah[0]['nsites']))
                # print(blah[1].shape)
                # print(sess.run(tf.shape(next_kd_batch_images)[0], feed_dict=train_feed_dict))
                # print(blah[2].shape)
                # print(blah[3].shape)
                # # print(sess.run(_pred_logfc_normed, feed_dict=train_feed_dict).shape)
                # # print(sess.run(_repression_y_normed, feed_dict=train_feed_dict).shape)
                # break

                train_evals = sess.run([
                    next_ka_batch_labels,  #0
                    _pred_biochem,  #1
                    next_tpm_batch,  #2
                    _repression_y_normed,  #3
                    _pred_logfc_normed,  #4
                    # _train_step,  #5
                ], feed_dict=train_feed_dict)

                print(train_evals[0].shape, train_evals[1].shape)
                print(train_evals[2]['images'].shape)
                print(train_evals[2]['labels'].shape)
                print(train_evals[2]['nsites'].shape)
                print(train_evals[3].shape, train_evals[4].shape)

                val_evals = sess.run([
                    next_ka_batch_labels,
                    _pred_biochem,
                    # next_tpm_batch,
                    # _repression_y_val_normed,
                    # _pred_logfc_val_normed
                ], feed_dict=val_feed_dict)

                # print(val_evals[0].shape, val_evals[1].shape)
                # print(val_evals[2]['images'].shape)
                # print(val_evals[2]['labels'].shape)
                # print(val_evals[2]['nsites'].shape)
                # print(val_evals[3].shape, val_evals[4].shape)

                sess.run(_train_step, feed_dict=train_feed_dict)
                # print(sess.run(_grads, feed_dict=train_feed_dict))
                # print(sess.run(_freeAGO_mean))

                break

            time_start = time.time()
            while True:
                try:
                    evals = sess.run([
                        next_ka_batch_labels,
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

                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(evals[1].flatten(), evals[0].flatten())
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

                    logfile.write('Time for epoch: {}\n'.format(time.time() - time_start))
                    logfile.write('Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(current_epoch, evals[5], evals[6], evals[7], evals[8]))
                    print('Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(current_epoch, evals[5], evals[6], evals[7], evals[8]))

                    val_evals = sess.run([
                        next_ka_batch_labels,
                        _pred_biochem,
                        _repression_y_val_normed,
                        _pred_logfc_val_normed
                    ], feed_dict=val_feed_dict)

                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(val_evals[1].flatten(), val_evals[0].flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'val_ka_scatter.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7, 7))
                    for ix in range(val_evals[2].shape[0]):
                        plt.scatter(val_evals[3][ix, :], val_evals[2][ix, :], color=COLORS)
                    plt.savefig(os.path.join(options.LOGDIR, 'val_tpm_scatter.png'))
                    plt.close()

                    break

    logfile.close()

        # feed_dict = {_phase_train: True, _keep_prob: 0.5, _handle: kd_train_handle}
        # # results = sess.run([_utr_ka_values, next_tpm_batch['freeAGOs'], _nbound, _all_feats], feed_dict=feed_dict)
        # # for x in results:
        # #     print(x.shape)

        # batch_mirs, batch, preds = sess.run([next_kd_batch_mirs, next_tpm_batch, _pred_logfc], feed_dict=feed_dict)
        # print(batch_mirs)
        # print(batch['transcripts'])
        # print(batch['nsites_mir'].astype(int).reshape([-1, NUM_TRAIN_GUIDES]))
        # print((preds > 0.001).astype(int))

        # feed_dict = {_phase_train: True, _keep_prob: 0.5, _handle: kd_val_handle}
        # print(sess.run(next_kd_batch_mirs, feed_dict=feed_dict))
