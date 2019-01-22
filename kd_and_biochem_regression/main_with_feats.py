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
    else:
        if options.VAL_MIR not in ALL_GUIDES:
            raise ValueError('Test miRNA not in mirseqs file.')
        TRAIN_GUIDES = [m for m in ALL_GUIDES if m != options.VAL_MIR]

    NUM_TRAIN = len(TRAIN_GUIDES)

    if options.PASSENGER:
        TRAIN_MIRS = np.array(list(zip(TRAIN_GUIDES, [x + '*' for x in TRAIN_GUIDES]))).flatten().tolist()
        ALL_MIRS = np.array(list(zip(ALL_GUIDES, [x + '*' for x in ALL_GUIDES]))).flatten().tolist()
    else:
        TRAIN_MIRS = TRAIN_GUIDES
        ALL_MIRS = ALL_GUIDES

    print("Repression datasets for training: {}".format(TRAIN_GUIDES))
    print("Repression miRNA considered for training: {}".format(TRAIN_MIRS))

    # create freeAGO concentration variables
    _freeAGO_mean = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(-4.0))
    _freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset', shape=[NUM_TRAIN, 1],
        initializer=tf.constant_initializer(0.0))

    if options.PASSENGER:
        _freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset', shape=[NUM_TRAIN, 1],
            initializer=tf.constant_initializer(-1.0))
        _freeAGO_all = tf.reshape(tf.concat([_freeAGO_guide_offset + _freeAGO_mean, _freeAGO_pass_offset + _freeAGO_mean], axis=1),
            [NUM_TRAIN * 2], name='freeAGO_all')
    else:
        _freeAGO_all = tf.squeeze(_freeAGO_guide_offset + _freeAGO_mean, name='freeAGO_all')

    # TPM data reader
    tpm_dataset = tf.data.TFRecordDataset(options.TPM_TFRECORDS)
    tpm_dataset = tpm_dataset.shuffle(buffer_size=1000)

    def _parse_fn(x):
        return parse_data_utils._parse_repression_function(x, TRAIN_MIRS, ALL_MIRS, options.MIRLEN, SEQLEN, options.NUM_FEATS, _freeAGO_all)

    tpm_dataset = tpm_dataset.map(_parse_fn)
    tpm_iterator = tpm_dataset.make_initializable_iterator()

    # build tpm batch
    next_tpm_batch = parse_data_utils._build_tpm_batch(tpm_iterator, options.REPRESSION_BATCH_SIZE, options.PASSENGER, NUM_TRAIN)

    if options.VAL_MIR in MIRNA_DATA_WITH_RBNS['mir'].values:
        TRAIN_MIRS_KDS = [x for x in MIRNA_DATA_WITH_RBNS['mir'].values if x != options.VAL_MIR]
        VAL_MIRS_KDS = [options.VAL_MIR]
    else:
        TRAIN_MIRS_KDS = MIRNA_DATA_WITH_RBNS['mir'].values
        VAL_MIRS_KDS = []

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
    _handle = tf.placeholder(tf.string, shape=[])
    kd_iterator = tf.data.Iterator.from_string_handle(_handle, kd_train_dataset.output_types)
    next_kd_batch_mirs, next_kd_batch_images, next_kd_batch_labels = kd_iterator.get_next()

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
        _pred_biochem = _pred_ka_values[-1 * options.KD_BATCH_SIZE:, :]
        _ka_loss = (tf.nn.l2_loss(tf.subtract(_pred_biochem, next_ka_batch_labels))) / options.KD_BATCH_SIZE
        _utr_ka_values = _pred_ka_values[:-1 * options.KD_BATCH_SIZE, :]

    # get logfc prediction
    _nbound = tf.sigmoid(_utr_ka_values + next_tpm_batch['freeAGOs'])
    _all_feats = tf.concat([_nbound, next_tpm_batch['features']], axis=1)
    _pred_logfc_ind_sites = tf.nn.relu(tf.squeeze(tf.matmul(_all_feats, _ts7_weights)))
    _pred_logfc_splits = tf.split(_pred_logfc_ind_sites, next_tpm_batch['nsites_mir'])
    _pred_logfc = tf.reshape(tf.stack([tf.reduce_sum(x) for x in _pred_logfc_splits]), [options.REPRESSION_BATCH_SIZE, NUM_TRAIN], name='pred_logfc')

    print('pred_logfc: {}'.format(_pred_logfc))

    # get repression loss
    if options.LOSS_TYPE == 'MEAN_CENTER':
        _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1, 1])
        _repression_y_normed = next_tpm_batch['labels'] - tf.reshape(tf.reduce_mean(next_tpm_batch['labels'], axis=1), [-1, 1])
    else:
        _pred_logfc_normed = _pred_logfc
        _repression_y_normed = next_tpm_batch['labels']

    print(_pred_logfc_normed)
    print(_repression_y_normed)
    _repression_loss = tf.nn.l2_loss(tf.subtract(_pred_logfc_normed, _repression_y_normed)) / (options.REPRESSION_BATCH_SIZE * NUM_TRAIN)

    # define regularizer
    _weight_regularize = tf.multiply(
        tf.nn.l2_loss(_freeAGO_guide_offset) +
        tf.nn.l2_loss(_cnn_weights['w1']) +
        tf.nn.l2_loss(_cnn_weights['w2']) +
        tf.nn.l2_loss(_cnn_weights['w3']) +
        tf.nn.l2_loss(_cnn_weights['w4']),
        options.LAMBDA
    )

    # define loss and train_step
    _loss = _ka_loss + _repression_loss + _weight_regularize

    _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(_update_ops):
        _train_step = tf.train.AdamOptimizer(options.LEARNING_RATE).minimize(_loss)

    # make model saver
    saver = tf.train.Saver(max_to_keep=options.NUM_EPOCHS + 1)

    if options.DRY_RUN:
        sys.exit()

    logfile = open(os.path.join(options.LOGDIR, 'out.log'), 'w', -1)

    # train model
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())



        for _ in range(options.NUM_EPOCHS):
            sess.run(tpm_iterator.initializer)
            sess.run(kd_train_iterator.initializer)
            sess.run(kd_val_iterator.initializer)

            # get training and validation handles for KD data
            kd_train_handle = sess.run(kd_train_iterator.string_handle())
            kd_val_handle = sess.run(kd_val_iterator.string_handle())

            train_feed_dict = {_phase_train: True, _keep_prob: 0.5, _handle: kd_train_handle}
            val_feed_dict = {_phase_train: False, _keep_prob: 1.0, _handle: kd_val_handle}

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
                    ], feed_dict=train_feed_dict)

                except tf.errors.OutOfRangeError:

                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(evals[0].flatten(), evals[1].flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'train_ka_scatter.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(evals[2].flatten(), evals[3].flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'train_repression_scatter.png'))
                    plt.close()

                    losses.append(evals[5])
                    fig = plt.figure(figsize=(7, 5))
                    plt.plot(losses)
                    plt.savefig(os.path.join(options.LOGDIR, 'train_losses.png'))
                    plt.close()

                    logfile.write('Time for epoch: {}\n'.format(time.time() - time_start))
                    logfile.write('Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(num_epochs, evals[5], evals[6], evals[7], evals[8]))

                    kd_val_batch = sess.run([next_kd_batch_mirs, next_kd_batch_images, next_kd_batch_labels], feed_dict=val_feed_dict)
                    temp_feed_dict = {
                        _phase_train: False, _keep_prob: 1.0, _handle: kd_val_handle,
                        _combined_x: kd_val_batch[1]
                    }
                    evals = sess.run(_pred_ka_values, feed_dict=temp_feed_dict)

                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(evals.flatten(), -1 * kd_val_batch[2].flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'val_ka_scatter.png'))
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
        # print(batch['nsites_mir'].astype(int).reshape([-1, NUM_TRAIN]))
        # print((preds > 0.001).astype(int))

        # feed_dict = {_phase_train: True, _keep_prob: 0.5, _handle: kd_val_handle}
        # print(sess.run(next_kd_batch_mirs, feed_dict=feed_dict))
