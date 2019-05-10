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
    parser.add_option("-k", "--kdfile", dest="KD_FILE", help="kd data")
    parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    parser.add_option("-m", "--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--test_mirna", dest="TEST_MIRNA", help="testing miRNA")
    # parser.add_option("--baseline", dest="BASELINE_METHOD", help="which baseline to use")
    parser.add_option("--loss_type", dest="LOSS_TYPE", help="which loss strategy", default='MEAN_CENTER')
    parser.add_option("--load_model", dest="LOAD_MODEL", help="if supplied, load latest model from this directory", default=None)
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs")

    (options, args) = parser.parse_args()

    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    ### READ miRNA DATA ###
    MIRNAS = pd.read_csv(options.MIR_SEQS, sep='\t')
    ALL_MIRS = list(MIRNAS['mir'].values)

    # split miRNAs into training and testing
    if options.TEST_MIRNA == 'none':
        TRAIN_MIRS = ALL_MIRS
        TEST_MIRS = ['mir139']
    else:
        if options.TEST_MIRNA not in ALL_MIRS:
            raise ValueError('Test miRNA not in mirseqs file.')
        TRAIN_MIRS = [m for m in ALL_MIRS if m != options.TEST_MIRNA]
        TEST_MIRS = [options.TEST_MIRNA]

    MIR_DICT = {}
    for row in MIRNAS.iterrows():
        guide_seq = row[1]['guide_seq']
        pass_seq = row[1]['guide_seq']
        MIR_DICT[row[1]['mir']] = {
            'mirseq': guide_seq,
            'site8': helpers.rev_comp(guide_seq[1:8]) + 'A',
            'one_hot': helpers.one_hot_encode(guide_seq[:config.MIRLEN])
        }
        MIR_DICT[row[1]['mir'] + '*'] = {
            'mirseq': pass_seq,
            'site8': helpers.rev_comp(pass_seq[1:8]) + 'A',
            'one_hot': helpers.one_hot_encode(pass_seq[:config.MIRLEN])
        }

    ### READ EXPRESSION DATA ###
    tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)#.dropna(subset=[options.BASELINE_METHOD])
    for mir in ALL_MIRS:
        if mir not in tpm.columns:
            raise ValueError('{} given in mirseqs file but not in tpm file.'.format(mir))

    train_tpm = tpm[TRAIN_MIRS + ['sequence', 'utr_length', 'orf_length']]
    test_tpm = tpm[TEST_MIRS + ['sequence', 'utr_length', 'orf_length']]

    # create data object
    repression_train_data = data_objects.RepressionData(train_tpm)
    repression_train_data.shuffle()
    repression_train_data.get_seqs(TRAIN_MIRS, config.OVERLAP_DIST, config.ONLY_CANON, options.RNAPLFOLD_FOLDER)


    def encode_seq_pairs(mirseq_one_hots, siteseqs, lookup):
        encoded = np.empty([len(siteseqs), 4 * config.MIRLEN, 4 * config.SEQLEN], 'float')

        if len(mirseq_one_hots) == 1:
            if lookup:
                mirseq_one_hot = MIR_ONE_HOT_DICT[mirseq_one_hots[0]]
            else:
                mirseq_one_hot = mirseq_one_hots[0]
            for ix, seq in enumerate(siteseqs):
                seq_one_hot = helpers.one_hot_encode(seq, SEQ_NT_DICT, TARGETS)
                encoded[ix, :, :] = np.outer(mirseq_one_hot, seq_one_hot)

        elif lookup:
            for ix, (mir, seq) in enumerate(zip(mirseq_one_hots, siteseqs)):
                mirseq_one_hot = MIR_ONE_HOT_DICT[mir]
                seq_one_hot = helpers.one_hot_encode(seq, SEQ_NT_DICT, TARGETS)
                encoded[ix, :, :] = np.outer(mirseq_one_hot, seq_one_hot)

        else:
            for ix, (mirseq_one_hot, seq) in enumerate(zip(mirseq_one_hots, siteseqs)):
                seq_one_hot = helpers.one_hot_encode(seq, SEQ_NT_DICT, TARGETS)
                encoded[ix, :, :] = np.outer(mirseq_one_hot, seq_one_hot)

        return encoded

    # split miRNAs into training and testing
    if options.TEST_MIRNA == 'none':
        TRAIN_MIRS = MIRS
        test_mirs = ['mir139']
        TEST_MIRNA = 'mir139'
    else:
        assert options.TEST_MIRNA in (MIRS + ['let7'])
        TEST_MIRNA = options.TEST_MIRNA
        TRAIN_MIRS = [m for m in MIRS if m != TEST_MIRNA]
        test_mirs = [TEST_MIRNA]

    print('Train miRNAs: {}'.format(TRAIN_MIRS))
    print('Test miRNAs: {}'.format(test_mirs))
    NUM_TRAIN = len(TRAIN_MIRS)
    NUM_TEST = len(test_mirs)

    print(NUM_TRAIN, NUM_TEST)

    # split TPM data into training and testing
    train_tpm = tpm[TRAIN_MIRS + ['sequence', 'utr_length', 'orf_length']]#.dropna(subset=[options.BASELINE_METHOD])
    test_tpm = tpm[test_mirs + ['sequence', 'utr_length', 'orf_length']]

    print(train_tpm.head())

    test_sitem8 = helpers.rev_comp(config.MIRSEQ_DICT[TEST_MIRNA][1:8])
    test_tpm['num_canon'] = [helpers.count_num_canon(utr, test_sitem8) for utr in test_tpm['sequence']]

    ### READ KD DATA ###
    data = pd.read_csv(options.KD_FILE, sep='\t')
    print(data.head())

    color_dict = {
        '8mer': 'red',
        '7mer-m8': 'orange',
        '7mer-a1': 'yellow',
        '6mer': 'green',
        '6mer-m8': 'blue',
        '6mer-a1': 'purple',
        'no site': 'grey'
    }

    # if only training on canonical sites, filter out nosite 12mers
    if config.ONLY_CANON:
        data = data[data['aligned_stype'] != 'no site']
        data['keep_prob'] = 1.0

    # otherwise, balance data to increase proportion of high affinity sites
    else:
        print("Balancing data...")
        data['nearest'] = np.minimum(0, np.round(data['log_kd'] * 4) / 4)
        data['count'] = 1
        temp = data.groupby('nearest').agg({'count': np.sum})
        temp['target'] = np.exp(temp.index + 5) * 500
        temp['keep_prob'] = np.minimum(1.0, temp['target'] / temp['count'])
        temp['keep_prob'] = [1.0 if x < -3 else y for (x, y) in zip(temp.index, temp['keep_prob'])]
        temp_dict = {x: y for (x, y) in zip(temp.index, temp['keep_prob'])}
        data['keep_prob'] = [temp_dict[x] for x in data['nearest']]

        data = data.drop(['nearest', 'count'], 1)

    # get rid of 12mers with sites in other registers
    temp = []
    print("Length of KD data before removing sites in other registers: {}".format(len(data)))
    # for mir, group in data.groupby('mir'):
    #     site8 = helpers.rev_comp(config.MIRSEQ_DICT[mir][1: 8]) + 'A'
    #     group = group[[helpers.best_stype_match(seq, site8) for seq in group['12mer']]]
    #     temp.append(group)

    # data = pd.concat(temp)
    data = data[data['best_stype'] == data['aligned_stype']]
    print("Length of KD data after removing sites in other registers: {}".format(len(data)))

    data['log_ka'] = np.maximum(0, (-1.0 * data['log_kd']))
    data['mirseq'] = [config.MIRSEQ_DICT_MIRLEN[mir] for mir in data['mir']]
    data['color'] = [color_dict[stype] for stype in data['aligned_stype']]

    if TEST_MIRNA in data['mir'].values:
        print('Plotting KD predictions with {}'.format(TEST_MIRNA))
        data_test = data[data['mir'] == TEST_MIRNA]
        data_test['keep_prob'] /= 10
    else:
        print('Plotting KD predictions with all 6 RBNS miRNAs')
        data_test = data.copy()
        data_test['keep_prob'] /= 60

    print(len(data_test))
    data_test = data_test[[np.random.random() < x for x in data_test['keep_prob']]]
    print("Test KD miRNAs:")
    print(data_test['mir'].unique())
    print(len(data_test))

    # test_kds_combined_x_old = np.zeros([len(data_test), 4 * config.MIRLEN, 4 * config.SEQLEN])
    # for i, row in enumerate(data_test.iterrows()):
    #     mirseq_one_hot = MIR_ONE_HOT_DICT[row[1]['mir']]
    #     seq_one_hot = helpers.one_hot_encode(row[1]['12mer'], SEQ_NT_DICT, TARGETS)
    #     test_kds_combined_x_old[i, :, :] = np.outer(mirseq_one_hot, seq_one_hot)

    # one-hot encode test 12mers
    test_kds_combined_x = encode_seq_pairs(data_test['mir'].values, data_test['12mer'].values, True)
    test_kds_labels = data_test['log_ka'].values
    test_kds_colors = data_test['color'].values

    # print(np.sum(np.abs(test_kds_combined_x - test_kds_combined_x_old)))

    data = data[~data['mir'].isin(test_mirs)]
    print("Length of KD training set: {}".format(len(data)))

    # one-hot encode a subset of the training 12mers
    data_test2 = data[[np.random.random() < x for x in (data['keep_prob'] / 60.0)]]
    print("Length of KD training set for plotting: {}".format(len(data_test2)))

    test_kds_combined_x2 = encode_seq_pairs(data_test2['mir'].values, data_test2['12mer'].values, True)
    test_kds_labels2 = data_test2['log_ka'].values
    test_kds_colors2 = data_test2['color'].values

    # create data object
    biochem_train_data = data_objects.BiochemData(data)
    biochem_train_data.shuffle()

    # make data objects for repression training data
    repression_train_data = data_objects.RepressionData(train_tpm)
    repression_train_data.shuffle()
    repression_train_data.get_seqs(TRAIN_MIRS, config.OVERLAP_DIST, config.ONLY_CANON, options.RNAPLFOLD_FOLDER)

    repression_test_data = data_objects.RepressionData(test_tpm)
    repression_test_data.get_seqs(test_mirs, config.OVERLAP_DIST, config.ONLY_CANON, options.RNAPLFOLD_FOLDER)

    sys.stdout.flush()

    sys.exit()

    ### DEFINE MODEL ###

    # reset and build the neural network
    tf.reset_default_graph()

    # start session
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:

        # create placeholders for input data
        _keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        _phase_train = tf.placeholder(tf.bool, name='phase_train')
        _combined_x = tf.placeholder(tf.float32, shape=[None, 4 * config.MIRLEN, 4 * config.SEQLEN], name='biochem_x')

        # make variable placeholders for training
        _biochem_y = tf.placeholder(tf.float32, shape=[None, 1], name='biochem_y')
        _utr_len = tf.placeholder(tf.float32, shape=[None, 1], name='utr_len')
        _repression_y = tf.placeholder(tf.float32, shape=[None, None], name='repression_y')

        # reshape, zero-center input
        _combined_x_4D = tf.expand_dims((_combined_x * 4.0) - 0.25, axis=3)

        # add layers for predicting KA
        _pred_ka_values, _cnn_weights = tf_helpers.seq2ka_predictor(_combined_x_4D, _keep_prob, _phase_train)

        biochem_saver = tf.train.Saver()

        # split data into biochem and repression and get biochem loss
        if config.BATCH_SIZE_BIOCHEM == 0:
            _pred_biochem = tf.constant(np.array([[0]]))
            _biochem_loss = tf.constant(0.0)
            _utr_ka_values = tf.reshape(_pred_ka_values, [-1])
        else:
            _pred_biochem = _pred_ka_values[-1 * config.BATCH_SIZE_BIOCHEM:, :]
            # _biochem_loss = (tf.nn.l2_loss(tf.subtract(_pred_biochem, _biochem_y) * tf.sqrt(_biochem_y))) * config.BIOCHEM_WEIGHT
            _biochem_loss = (tf.nn.l2_loss(tf.subtract(_pred_biochem, _biochem_y))) * config.BIOCHEM_WEIGHT
            _utr_ka_values = tf.reshape(_pred_ka_values[:-1 * config.BATCH_SIZE_BIOCHEM, :], [-1])

        # reshape repression ka values
        _utr_max_size = tf.placeholder(tf.int32, shape=[], name='utr_max_size')
        _utr_split_sizes = tf.placeholder(tf.int32, shape=[config.BATCH_SIZE_REPRESSION * NUM_TRAIN * 2], name='utr_split_sizes')
        _utr_ka_values_reshaped = tf_helpers.pad_kd_from_genes(_utr_ka_values, _utr_split_sizes, _utr_max_size, NUM_TRAIN, config.BATCH_SIZE_REPRESSION)

        print('utr_ka_reshaped: {}'.format(_utr_ka_values_reshaped))

        # get repression prediction
        init_params = [
            config.FREEAGO_INIT,
            config.GUIDE_OFFSET_INIT,
            config.PASS_OFFSET_INIT,
            config.DECAY_INIT,
            config.UTR_COEF_INIT
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

        # get repression loss
        if options.LOSS_TYPE == 'MEAN_CENTER':
            _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1, 1])
            _repression_y_normed = _repression_y - tf.reshape(tf.reduce_mean(_repression_y, axis=1), [-1, 1])
            _repression_loss = (config.REPRESSION_WEIGHT * tf.nn.l2_loss(tf.subtract(_pred_logfc_normed, _repression_y_normed))) / config.BATCH_SIZE_REPRESSION

        else:
            _repression_loss = (config.REPRESSION_WEIGHT * tf.nn.l2_loss(tf.subtract(_pred_logfc, _repression_y))) / config.BATCH_SIZE_REPRESSION

        # define regularizer
        _weight_regularize = tf.multiply(
            tf.nn.l2_loss(_results['freeAGO_guide_offset']) +
            tf.nn.l2_loss(_cnn_weights['w1']) +
            tf.nn.l2_loss(_cnn_weights['w2']) +
            tf.nn.l2_loss(_cnn_weights['w3']) +
            tf.nn.l2_loss(_cnn_weights['w4']),
            config.LAMBDA
        )

        _loss = _biochem_loss + _repression_loss + _weight_regularize

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(config.STARTING_LEARNING_RATE, global_step,
                                                   2500, 0.90, staircase=True)

        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops):
            _train_step = tf.train.AdamOptimizer(config.STARTING_LEARNING_RATE).minimize(_loss, global_step=global_step)

        saver = tf.train.Saver(max_to_keep=config.NUM_EPOCHS + 1)

        ### TRAIN MODEL ###
        sess.run(tf.global_variables_initializer())

        if options.LOAD_MODEL is not None:
            latest = tf.train.latest_checkpoint(options.LOAD_MODEL)
            print('Restoring from {}'.format(latest))
            saver.restore(sess, latest)

        # plot layer1 weights
        conv_weights = sess.run(_cnn_weights['w1'])
        xlabels = list(SEQ_NTS)
        ylabels = list(MIR_NTS)
        helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1_init.pdf'))

        # plot importance matrix of nucleotide positions
        conv_weights = np.abs(sess.run(_cnn_weights['w3']))
        conv_weights = np.sum(conv_weights, axis=(2, 3))
        vmin, vmax = np.min(conv_weights), np.max(conv_weights)
        xlabels = ['s{}'.format(i + 1) for i in range(config.SEQLEN)]
        ylabels = ['m{}'.format(i + 1) for i in list(range(config.MIRLEN))[::-1]]
        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                    cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
        plt.savefig(os.path.join(options.LOGDIR, 'convolution3_init.pdf'))
        plt.close()

        print("Started training...")

        # things to record during training
        records = {
            'train_loss': [],
            'decay': [],
            'utr_coef': [],
            'freeAGO_mean': []
        }

        current_epoch = 0
        step = -1

        while current_epoch < config.NUM_EPOCHS:

            # get repression data batch
            batch_genes, next_epoch, all_seqs, all_feats, train_sizes, max_sites, batch_repression_y = repression_train_data.get_next_batch(config.BATCH_SIZE_REPRESSION, TRAIN_MIRS)

            if next_epoch:
                current_epoch += 1

            # if none of the genes have sites, continue
            if max_sites == 0:
                continue

            num_total_train_seqs = np.sum(train_sizes)
            batch_combined_x = np.zeros([num_total_train_seqs + config.BATCH_SIZE_BIOCHEM, 4 * config.MIRLEN, 4 * config.SEQLEN])
            batch_feats = np.zeros([num_total_train_seqs, NUM_TRAIN * 2, max_sites, config.NUM_TS7])

            # fill features for utr sites for both the guide and passenger strands
            mirlist = TRAIN_MIRS * config.BATCH_SIZE_REPRESSION
            current_ix = 0
            mir_ix = 0
            for mir, (seq_list_guide, seq_list_pass), (feats_guide, feats_pass) in zip(mirlist, all_seqs, all_feats):
                if len(seq_list_guide) > 0:
                    batch_combined_x[current_ix: current_ix + len(seq_list_guide), :, :] = encode_seq_pairs([mir], seq_list_guide, True)
                    for temp_ix, feat_list in enumerate(feats_guide):
                        batch_feats[current_ix: mir_ix, temp_ix, :] = feat_list
                    current_ix += len(seq_list_guide)
                    mir_ix += 1

                if len(seq_list_pass) > 0:
                    batch_combined_x[current_ix: current_ix + len(seq_list_pass), :, :] = encode_seq_pairs([mir + '*'], seq_list_pass, True)
                    for temp_ix, feat_list in enumerate(feats_pass):
                        batch_feats[current_ix: mir_ix, temp_ix, :] = feat_list
                    current_ix += len(seq_list_pass)
                    mir_ix += 1

            if config.BATCH_SIZE_BIOCHEM > 0:

                _, biochem_train_batch = biochem_train_data.get_next_batch(config.BATCH_SIZE_BIOCHEM - 2)

                batch_biochem_y = np.expand_dims(np.array(list(biochem_train_batch['log_ka'].values) + [0, 0]), 1)
                mirseq_one_hots = [MIR_ONE_HOT_DICT[x] for x in biochem_train_batch['mir']]
                siteseqs = list(biochem_train_batch['12mer'].values)

                # add nonmatching sequence for miRNA
                mirseq_one_hots.append(MIR_ONE_HOT_DICT[mir])
                target_no_match = helpers.get_target_no_match(config.MIRSEQ_DICT[mir], config.SEQLEN)
                siteseqs.append(target_no_match)

                # add nonmatching sequence for a random miRNA sequence
                random_mir = helpers.generate_random_seq(config.MIRLEN)
                random_target = helpers.get_target_no_match(random_mir, config.SEQLEN)
                mirseq_one_hots.append(helpers.one_hot_encode(random_mir[::-1], MIR_NT_DICT, TARGETS))
                siteseqs.append(random_target)

                batch_combined_x[current_ix:, :, :] = encode_seq_pairs(mirseq_one_hots, siteseqs, False)

            else:
                batch_biochem_y = np.array([[0]])

            # define y values
            y_vals = batch_repression_y

            # make feed dict for training
            feed_dict = {
                _keep_prob: config.KEEP_PROB_TRAIN,
                _phase_train: True,
                _combined_x: batch_combined_x,
                _biochem_y: batch_biochem_y,
                _utr_max_size: max_sites,
                _utr_split_sizes: train_sizes,
                _repression_y: y_vals,
                _utr_len: tpm.loc[batch_genes][['utr_length']].values
            }

            # run train step
            _, l1, l2, l3, train_loss = sess.run([_train_step, _biochem_loss, _repression_loss,
                                                      _weight_regularize, _loss], feed_dict=feed_dict)

            step += 1
            if next_epoch or (step == 0):

                feed_dict = {
                    _keep_prob: 1.0,
                    _phase_train: False,
                    _combined_x: batch_combined_x,
                    _biochem_y: batch_biochem_y,
                    _utr_max_size: max_sites,
                    _utr_split_sizes: train_sizes,
                    _repression_y: y_vals,
                    _utr_len: tpm.loc[batch_genes][['utr_length']].values
                }

                # save model
                saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=current_epoch)

                # calculate and plot train performance
                print('Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(current_epoch, train_loss, l1, l2, l3))
                records['train_loss'].append(train_loss)

                fig = plt.figure(figsize=(7, 5))
                plt.plot(records['train_loss'])
                plt.savefig(os.path.join(options.LOGDIR, 'train_losses.png'))
                plt.close()

                if config.BATCH_SIZE_BIOCHEM > 0:
                    train_biochem_preds = sess.run(_pred_biochem, feed_dict=feed_dict)

                    fig = plt.figure(figsize=(7, 7))
                    plt.scatter(train_biochem_preds.flatten(), batch_biochem_y.flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'train_biochem_scatter.png'))
                    plt.close()

                train_repression_kds = sess.run(_utr_ka_values, feed_dict=feed_dict)

                fig = plt.figure(figsize=(7, 7))
                plt.hist(train_repression_kds.flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'train_repression_kds_hist.png'))
                plt.close()

                if options.LOSS_TYPE == 'MEAN_CENTER':
                    train_repression_preds, train_repression_ys = sess.run([_pred_logfc_normed, _repression_y_normed],
                                                                            feed_dict=feed_dict)
                else:
                    train_repression_preds, train_repression_ys = sess.run([_pred_logfc, _repression_y],
                                                                            feed_dict=feed_dict)

                fig = plt.figure(figsize=(7, 7))
                for i in range(config.BATCH_SIZE_REPRESSION):
                    temp = pd.DataFrame({'pred': train_repression_preds[i, :], 'actual': train_repression_ys[i, :]})
                    temp = temp.sort_values('pred')
                    plt.plot(temp['pred'], temp['actual'])

                plt.savefig(os.path.join(options.LOGDIR, 'train_repression_fits.png'))
                plt.close()

                # plot weights
                conv_weights = sess.run(_cnn_weights['w1'])
                xlabels = ['U', 'A', 'G', 'C']
                ylabels = ['A', 'U', 'C', 'G']
                helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1.pdf'))

                # plot importance matrix
                conv_weights = np.abs(sess.run(_cnn_weights['w3']))
                conv_weights = np.sum(conv_weights, axis=(2, 3))
                vmin, vmax = np.min(conv_weights), np.max(conv_weights)
                xlabels = ['s{}'.format(i + 1) for i in range(config.SEQLEN)]
                ylabels = ['m{}'.format(i + 1) for i in list(range(config.MIRLEN))[::-1]]
                fig = plt.figure(figsize=(4, 4))
                sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                            cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
                plt.savefig(os.path.join(options.LOGDIR, 'convolution3.pdf'))
                plt.close()

                feed_dict = {
                    _keep_prob: 1.0,
                    _phase_train: False,
                    _combined_x: test_kds_combined_x
                }

                pred_test_kds = sess.run(_pred_ka_values, feed_dict=feed_dict)

                fig = plt.figure(figsize=(7, 7))
                plt.scatter(pred_test_kds, test_kds_labels, s=20, c=test_kds_colors)
                plt.title(TEST_MIRNA)
                plt.savefig(os.path.join(options.LOGDIR, 'test_kd_scatter.png'))
                plt.close()

                feed_dict = {
                    _keep_prob: 1.0,
                    _phase_train: False,
                    _combined_x: test_kds_combined_x2
                }

                pred_test_kds2 = sess.run(_pred_ka_values, feed_dict=feed_dict)

                fig = plt.figure(figsize=(7, 7))
                plt.scatter(pred_test_kds2, test_kds_labels2, s=20, c=test_kds_colors2)
                plt.title('train miRNAs')
                plt.savefig(os.path.join(options.LOGDIR, 'train_kd_scatter.png'))
                plt.close()

                current_decay = sess.run(_results['decay'])
                current_utr_coef = sess.run(_results['utr_coef'])

                current_freeAGO = sess.run(_results['freeAGO_all'])
                current_freeAGO_mean = sess.run(_results['freeAGO_mean'])
                current_freeAgo_guide_offset = np.mean(sess.run(_results['freeAGO_guide_offset']))
                current_freeAgo_pass_offset = np.mean(sess.run(_results['freeAGO_pass_offset']))

                records['decay'].append(current_decay)
                records['utr_coef'].append(current_utr_coef)
                records['freeAGO_mean'].append(current_freeAGO_mean)

                fig = plt.figure(figsize=(10, 5))
                plt.plot(records['decay'])
                plt.savefig(os.path.join(options.LOGDIR, 'decay_fits.png'))
                plt.close()

                fig = plt.figure(figsize=(10, 5))
                plt.plot(records['utr_coef'])
                plt.savefig(os.path.join(options.LOGDIR, 'utr_coef_fits.png'))
                plt.close()

                fig = plt.figure(figsize=(10, 5))
                plt.plot(records['freeAGO_mean'])
                plt.savefig(os.path.join(options.LOGDIR, 'freeAGO_mean_fits.png'))
                plt.close()

        # after all training, write results to file
        current_freeAGO = current_freeAGO.reshape([NUM_TRAIN, 2])
        freeAGO_df = pd.DataFrame({'mir': TRAIN_MIRS,
                                   'guide': current_freeAGO[:, 0],
                                   'passenger': current_freeAGO[:, 1]})

        freeAGO_df.to_csv(os.path.join(options.LOGDIR, 'freeAGO_final.txt'), sep='\t', index=False)
        with open(os.path.join(options.LOGDIR, 'fitted_params.txt'), 'w') as outfile:
            outfile.write('decay\t{}\n'.format(current_decay))
            outfile.write('utr_coef\t{}\n'.format(current_utr_coef))
