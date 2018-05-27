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

import config, helpers, data_objects_endog

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-k", "--kdfile", dest="KD_FILE", help="kd data")
    parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    parser.add_option("-m", "--mirna", dest="TEST_MIRNA", help="testing miRNA")
    parser.add_option("--let7_sites", dest="LET7_SITES", help="let-7a site kds")
    parser.add_option("--let7_mask", dest="LET7_MASK", help="let-7a site mask")
    parser.add_option("-p", "--prefit", dest="PREFIT", help="prefit xs and ys")
    parser.add_option("--hidden1", dest="HIDDEN1", type=int, help="number of nodes in layer 1")
    parser.add_option("--hidden2", dest="HIDDEN2", type=int, help="number of nodes in layer 2")
    parser.add_option("--hidden3", dest="HIDDEN3", type=int, help="number of nodes in layer 3")
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("--pretrain", dest="PRETRAIN", help="pretrain directory")

    (options, args) = parser.parse_args()

    PRETRAIN_SAVE_PATH = os.path.join(options.LOGDIR, 'pretrain_saved')
    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    ### READ PREFIT DATA ###
    prefit = pd.read_csv(options.PREFIT, sep='\t', index_col=0)

    ### READ let-7 sites ###

    let7_sites = pd.read_csv(options.LET7_SITES, sep='\t', index_col=0)
    let7_mask = pd.read_csv(options.LET7_MASK, sep='\t', index_col=0)
    let7_num_kds = len(let7_sites.columns)

    ### READ EXPRESSION DATA ###
    tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)

    MIRS = [x for x in tpm.columns if ('mir' in x) or ('lsy' in x)]

    # split miRNAs into training and testing
    if options.TEST_MIRNA == 'none':
        train_mirs = MIRS
        test_mirs = ['mir139']
        TEST_MIRNA = 'mir139'
        # train_mirs = [m for m in MIRS if m != TEST_MIRNA]
    else:
        assert options.TEST_MIRNA in MIRS
        TEST_MIRNA = options.TEST_MIRNA
        train_mirs = [m for m in MIRS if m != TEST_MIRNA]
        test_mirs = [TEST_MIRNA]
        

    print('Train miRNAs: {}'.format(train_mirs))
    print('Test miRNAs: {}'.format(test_mirs))
    NUM_TRAIN = len(train_mirs)
    NUM_TEST = len(test_mirs)

    print(NUM_TRAIN, NUM_TEST)

    tpm = tpm.rename(columns={'sequence': 'Sequence'})

    # split tpm data into training and testing
    train_tpm = tpm[train_mirs + ['Sequence']]
    test_tpm = tpm[test_mirs + ['Sequence']]

    ### READ KD DATA ###
    data = pd.read_csv(options.KD_FILE, sep='\t')
    # data.columns = ['mir','mirseq_full','seq','log kd','stype']
    data.columns = ['seq','log_kd','mir','mirseq_full','stype']

    # zero-center and normalize Ka's
    data['keep_prob'] = (1 / (1 + np.exp(data['log_kd'] + 2)))
    data['log ka'] = (-1.0 * data['log_kd'])
    data['mirseq'] = [config.MIRSEQ_DICT_MIRLEN[mir] for mir in data['mir']]
    data['sitem8'] = [helpers.rev_comp(mirseq[1:8]) for mirseq in data['mirseq_full']]
    data['color'] = [helpers.get_color(sitem8, seq) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]
    data['color2'] = [helpers.get_color(sitem8, seq[2:10]) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

    # get rid of sequences with sites out of register
    print('Length of KD data: {}'.format(len(data)))
    data = data[data['color'] == data['color2']].drop('color2',1)
    print('Length of KD data, in register: {}'.format(len(data)))

    if TEST_MIRNA in data['mir'].values:
        print('Testing on {}'.format(TEST_MIRNA))
        data_test = data[data['mir'] == TEST_MIRNA]
        data_test['keep_prob'] /= 4
    else:
        print('Testing on all')
        data_test = data.copy()
        data_test['keep_prob'] /= 20

    data_test = data_test[[np.random.random() < x for x in data_test['keep_prob']]]
    print("Test KD miRNAs:")
    print(data_test['mir'].unique())
    print(len(data_test))

    test_kds_combined_x = np.zeros([len(data_test), 48, 48])
    for i, row in enumerate(data_test.iterrows()):
        mirseq_one_hot = config.ONE_HOT_DICT[row[1]['mir']]
        seq_one_hot = helpers.one_hot_encode(row[1]['seq'], config.SEQ_NT_DICT, config.TARGETS)
        test_kds_combined_x[i,:,:] = np.outer(mirseq_one_hot, seq_one_hot)
    
    test_kds_combined_x = np.expand_dims((test_kds_combined_x*4) - 0.25, 3)   
    test_kds_labels = data_test['log ka'].values + config.ZERO_OFFSET # OFFSET
    
    data = data[~data['mir'].isin(test_mirs)]
    print(len(data))

    # create data object
    biochem_train_data = data_objects_endog.BiochemData(data)
    biochem_train_data.shuffle()

    # make data objects for repression training data
    repression_train_data = data_objects_endog.RepressionData(train_tpm)
    repression_train_data.shuffle()
    repression_train_data.get_seqs(train_mirs)    

    ### DEFINE MODEL ###

    # reset and build the neural network
    tf.reset_default_graph()

    # start session
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:

        # create placeholders for input data
        _keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        _phase_train = tf.placeholder(tf.bool, name='phase_train')
        _combined_x = tf.placeholder(tf.float32, shape=[None, 4 * config.MIRLEN, 4 * config.SEQLEN, 1], name='biochem_x')
        _pretrain_y =  tf.placeholder(tf.float32, shape=[None, 1], name='pretrain_y')

        # add layer 1
        with tf.name_scope('layer1'):
            _w1, _b1 = helpers.get_conv_params(4, 4, 1, options.HIDDEN1, 'layer1')
            _preactivate1 = tf.nn.conv2d(_combined_x, _w1, strides=[1, 4, 4, 1], padding='VALID') + _b1

            _preactivate1_bn = tf.layers.batch_normalization(_preactivate1, axis=1, training=_phase_train)

            _layer1 = tf.nn.relu(_preactivate1_bn)

        # add layer 2
        with tf.name_scope('layer2'):
            _w2, _b2 = helpers.get_conv_params(2, 2, options.HIDDEN1, options.HIDDEN2, 'layer2')
            _preactivate2 = tf.nn.conv2d(_layer1, _w2, strides=[1, 1, 1, 1], padding='VALID') + _b2

            _preactivate2_bn = tf.layers.batch_normalization(_preactivate2, axis=1, training=_phase_train)

            _layer2 = tf.nn.relu(_preactivate2_bn)

            _dropout2 = tf.nn.dropout(_layer2, _keep_prob)

        # add layer 3
        with tf.name_scope('layer3'):
            _w3, _b3 = helpers.get_conv_params(config.MIRLEN-1, config.SEQLEN-1, options.HIDDEN2, options.HIDDEN3, 'layer3')
            _preactivate3 = tf.nn.conv2d(_dropout2, _w3, strides=[1, config.MIRLEN-1, config.SEQLEN-1, 1], padding='VALID') + _b3

            _preactivate3_bn = tf.layers.batch_normalization(_preactivate3, axis=1, training=_phase_train)

            _layer3 = tf.nn.relu(_preactivate3_bn)

        # # add layer 2.5
        # with tf.name_scope('layer2_5'):
        #     _w2_5, _b2_5 = helpers.get_conv_params(2, 2, options.HIDDEN2, options.HIDDEN2, 'layer2_5')
        #     _preactivate2_5 = tf.nn.conv2d(_layer2, _w2_5, strides=[1, 1, 1, 1], padding='VALID') + _b2_5

        #     _preactivate2_5_bn = tf.layers.batch_normalization(_preactivate2_5, axis=1, training=_phase_train)

        #     _layer2_5 = tf.nn.relu(_preactivate2_5_bn)

        # # add layer 3
        # with tf.name_scope('layer3'):
        #     _w3, _b3 = helpers.get_conv_params(config.MIRLEN-2, config.SEQLEN-2, options.HIDDEN2, options.HIDDEN3, 'layer3')
        #     _preactivate3 = tf.nn.conv2d(_layer2_5, _w3, strides=[1, config.MIRLEN-2, config.SEQLEN-2, 1], padding='VALID') + _b3

        #     _preactivate3_bn = tf.layers.batch_normalization(_preactivate3, axis=1, training=_phase_train)

        #     _layer3 = tf.nn.relu(_preactivate3_bn)

        # add dropout
        with tf.name_scope('dropout'):
            _dropout = tf.nn.dropout(_layer3, _keep_prob)

        # reshape to 1D tensor
        _layer_flat = tf.reshape(_dropout, [-1, options.HIDDEN3])

        # add last layer
        with tf.name_scope('final_layer'):
            with tf.name_scope('weights'):
                _w4 = tf.get_variable("final_layer_weight", shape=[options.HIDDEN3, 1],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                tf.add_to_collection('weight', _w4)
            with tf.name_scope('biases'):
                _b4 = tf.get_variable("final_layer_bias", shape=[1],
                                    initializer=tf.constant_initializer(0.0))
                tf.add_to_collection('bias', _b4)

            # apply final layer
            _pred_ind_values = (tf.matmul(_layer_flat, _w4) + _b4) * config.NORM_RATIO

        _pretrain_loss = tf.nn.l2_loss(tf.subtract(_pred_ind_values, _pretrain_y))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            _pretrain_step = tf.train.AdamOptimizer(config.STARTING_LEARNING_RATE).minimize(_pretrain_loss)

        saver_pretrain = tf.train.Saver()

        # make variable placeholders for training
        _repression_weight = tf.placeholder(tf.float32, name='repression_weight')
        _biochem_y = tf.placeholder(tf.float32, shape=[None, 1], name='biochem_y')
        _utr_len = tf.placeholder(tf.float32, shape=[None, 1], name='utr_len')
        _let7_sites = tf.placeholder(tf.float32, shape=[None, 1, None], name='let7_sites')
        _let7_mask = tf.placeholder(tf.float32, shape=[None, 1, None], name='let7_mask')
        _repression_max_size = tf.placeholder(tf.int32, shape=[], name='repression_max_size')
        _repression_split_sizes = tf.placeholder(tf.int32, shape=[config.BATCH_SIZE_REPRESSION*NUM_TRAIN*2], name='repression_split_sizes')
        _repression_y = tf.placeholder(tf.float32, shape=[None, None], name='repression_y')
        # _prefit_xs = tf.placeholder(tf.float32, shape=[None, None], name='prefit_xs')
        # _prefit_ys = tf.placeholder(tf.float32, shape=[None, None], name='prefit_ys')

        # construct global variables
        # freeAGO_init = np.zeros([1, NUM_TRAIN*2, 1])
        # freeAGO_init[0,:,0] = [0,-2]*NUM_TRAIN
        # print(freeAGO_init)
        # _freeAGO_all = tf.get_variable('freeAGO_all', shape=[1, NUM_TRAIN*2, 1],
        #                                 initializer=tf.constant_initializer(freeAGO_init))

        _freeAGO_init = tf.get_variable('freeAGO_init', shape=[1,2,1],
                                        initializer=tf.constant_initializer(np.array([-3,-4]).reshape([1,2,1])))
        _freeAGO_tile = tf.constant([1, NUM_TRAIN, 1])
        _freeAGO_all = tf.tile(_freeAGO_init, _freeAGO_tile)

        _decay = tf.get_variable('decay', shape=(), initializer=tf.constant_initializer(-0.6194858), trainable=False)

        # _utr_coef = tf.get_variable('utr_coef', shape=[1, NUM_TRAIN], initializer=tf.constant_initializer(-9))
        # _freeAGO_let7 = tf.get_variable('freeAGO_let7', shape=[1, NUM_TRAIN, 1],
        #                                 initializer=tf.constant_initializer(-8))

        _utr_coef = tf.get_variable('utr_coef', shape=(),
                                    initializer=tf.constant_initializer(-8.6379089), trainable=False)
        _freeAGO_let7 = tf.get_variable('freeAGO_let7', shape=[1, 1, 1],
                                        initializer=tf.constant_initializer(-7.48320436 + config.ZERO_OFFSET), trainable=False)

        # construct a mask based on the number of sites per gene
        _repression_mask = tf.reshape(tf.sequence_mask(_repression_split_sizes, dtype=tf.float32),
                                      [config.BATCH_SIZE_REPRESSION, NUM_TRAIN*2, -1])

        # get padding dimensions
        _repression_split_sizes_expand = tf.expand_dims(_repression_split_sizes, 1)
        _repression_paddings = tf.concat([tf.zeros(shape=tf.shape(_repression_split_sizes_expand), dtype=tf.int32),
                                          _repression_max_size - _repression_split_sizes_expand], axis=1)
        
        # split data into biochem and repression
        _pred_biochem = _pred_ind_values[-1 * config.BATCH_SIZE_BIOCHEM:, :]
        _pred_repression_flat = tf.reshape(_pred_ind_values[:-1 * config.BATCH_SIZE_BIOCHEM, :], [-1])
        # _pred_repression_flat = tf.reshape(_pred_ind_values, [-1])

        # split repression data and pad into config.BATCH_SIZE_BIOCHEM x NUM_TRAIN*2 x max_size matrix
        _pred_repression_splits = tf.split(_pred_repression_flat, _repression_split_sizes)
        _pred_repression_splits_padded = [tf.pad(_pred_repression_splits[ix], _repression_paddings[ix:ix+1,:]) for ix in range(config.BATCH_SIZE_REPRESSION*NUM_TRAIN*2)]
        _pred_repression_splits_padded_stacked = tf.stack(_pred_repression_splits_padded)
        _pred_repression = tf.reshape(_pred_repression_splits_padded_stacked, [config.BATCH_SIZE_REPRESSION, NUM_TRAIN*2, -1])

        # calculate predicted number bound and predicted log fold-change
        _pred_nbound_split = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO_all + _pred_repression), _repression_mask), axis=2)
        _pred_nbound = tf.reduce_sum(tf.reshape(_pred_nbound_split, [config.BATCH_SIZE_REPRESSION, NUM_TRAIN, 2]), axis=2)
        _pred_nbound_let7 = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO_let7 + _let7_sites), _let7_mask), axis=2)
        _pred_nbound_init = tf.exp(_utr_coef) * _utr_len

        _pred_nbound_total = _pred_nbound + _pred_nbound_let7 + _pred_nbound_init

        # _pred_logfc = tf.concat([-1.0 * tf.log1p(_pred_nbound_total / tf.exp(_decay)), _prefit_xs], axis=1)
        _pred_logfc = -1.0 * tf.log1p(_pred_nbound_total / tf.exp(_decay))
        _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1,1])

        # _repression_y_concat = tf.concat([_repression_y , _prefit_ys], axis=1)
        # _repression_y_normed = _repression_y_concat - tf.reshape(tf.reduce_mean(_repression_y_concat, axis=1), [-1,1])
        _repression_y_normed = _repression_y - tf.reshape(tf.reduce_mean(_repression_y, axis=1), [-1,1])

        _weight_regularize = tf.multiply(tf.nn.l2_loss(_w1) \
                                + tf.nn.l2_loss(_w2) \
                                + tf.nn.l2_loss(_w3) \
                                + tf.nn.l2_loss(_w4), config.LAMBDA)

        _biochem_loss = tf.nn.l2_loss(tf.subtract(_pred_biochem, _biochem_y))
        _repression_loss = _repression_weight * tf.nn.l2_loss(tf.subtract(_pred_logfc_normed, _repression_y_normed))

        _loss = _biochem_loss + _repression_loss + _weight_regularize

        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops):
            _train_step = tf.train.AdamOptimizer(config.STARTING_LEARNING_RATE).minimize(_loss)

        saver = tf.train.Saver(max_to_keep=config.NUM_EPOCHS+1)


        ### TRAIN MODEL ###

        sess.run(tf.global_variables_initializer())

        # options: 'pretrain', 'none'
        # PRETRAIN = '/lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_pretrain/pretrain_saved'
        # PRETRAIN = 'pretrain'
        # PRETRAIN = 'none'

        if options.PRETRAIN == 'pretrain':

            print("Started pretraining...")

            # plot weights
            conv_weights = sess.run(_w1)
            xlabels = ['U','A','G','C']
            ylabels = ['A','U','C','G']
            helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1_init.pdf'))

            conv_weights = np.abs(sess.run(_w3))
            conv_weights = np.sum(conv_weights, axis=(2,3))
            vmin, vmax = np.min(conv_weights), np.max(conv_weights)
            xlabels = ['s{}'.format(i+1) for i in range(config.SEQLEN)]
            ylabels = ['m{}'.format(i+1) for i in list(range(config.MIRLEN))[::-1]]
            fig = plt.figure(figsize=(4,4))
            sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                        cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
            plt.savefig(os.path.join(options.LOGDIR, 'convolution3_init.pdf'))
            plt.close()

            for i in range(1000):
                xs, ys = helpers.make_pretrain_data(200, config.MIRLEN, config.SEQLEN)
                feed_dict = {
                    _keep_prob: config.KEEP_PROB_TRAIN,
                    _phase_train: True,
                    _combined_x: xs,
                    _pretrain_y: ys
                }
                train_pred, _ = sess.run([_pred_ind_values, _pretrain_step], feed_dict=feed_dict)

                if ((i+1) % 200) == 0:
                    fig = plt.figure(figsize=(5,5))
                    plt.scatter(train_pred, ys, s=20)
                    plt.savefig(os.path.join(options.LOGDIR, 'pretrain_train_scatter.pdf'))
                    plt.close()

                    print(i+1)

            test_xs, test_ys = helpers.make_pretrain_data(200, config.MIRLEN, config.SEQLEN)
            feed_dict = {
                _keep_prob: 1.0,
                _phase_train: False,
                _combined_x: test_xs,
                _pretrain_y: test_ys
            }
            test_pred = sess.run(_pred_ind_values, feed_dict=feed_dict)

            fig = plt.figure(figsize=(5,5))
            plt.scatter(test_pred, test_ys, s=20)
            plt.savefig(os.path.join(options.LOGDIR, 'pretrain_test_scatter.pdf'))
            plt.close()

            # plot weights
            conv_weights = sess.run(_w1)
            xlabels = ['U','A','G','C']
            ylabels = ['A','U','C','G']
            helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1_pretrained.pdf'))

            conv_weights = np.abs(sess.run(_w3))
            conv_weights = np.sum(conv_weights, axis=(2,3))
            vmin, vmax = np.min(conv_weights), np.max(conv_weights)
            xlabels = ['s{}'.format(i+1) for i in range(config.SEQLEN)]
            ylabels = ['m{}'.format(i+1) for i in list(range(config.MIRLEN))[::-1]]
            fig = plt.figure(figsize=(4,4))
            sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                        cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
            plt.savefig(os.path.join(options.LOGDIR, 'convolution3_pretrained.pdf'))
            plt.close()

            # save pretrained model
            saver_pretrain.save(sess, os.path.join(PRETRAIN_SAVE_PATH, 'model'))

            print("Finished pretraining")
            sys.exit()

        elif options.PRETRAIN != 'none':
            latest = tf.train.latest_checkpoint(options.PRETRAIN)
            print('Restoring from {}'.format(latest))
            saver_pretrain.restore(sess, latest)

            # reset later layers
            sess.run(_w3.initializer)
            sess.run(_b3.initializer)
            sess.run(_w4.initializer)
            sess.run(_b4.initializer)


        print("Started training...")

        step_list = []
        train_losses = []
        test_losses = []
        last_batch = False

        step = -1
        current_epoch = 0

        # save initial model
        # saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=current_epoch)
        times, times2 = [], []

        while True:

            t0 = time.time()

            # get repression data batch
            batch_genes, next_epoch, all_seqs, train_sizes, max_sites, batch_repression_y = repression_train_data.get_next_batch2(config.BATCH_SIZE_REPRESSION, train_mirs)

            if next_epoch:
                current_epoch += 1
                # config.REPRESSION_WEIGHT += 0.2
                # if repression_train_data.num_epochs >= config.NUM_EPOCHS:
                #     last_batch = True

            # if none of the genes have sites, continue
            if max_sites == 0:
                continue

            # get biochem data batch
            _, biochem_train_batch = biochem_train_data.get_next_batch(config.BATCH_SIZE_BIOCHEM)

            num_total_train_seqs = np.sum(train_sizes)
            batch_combined_x = np.zeros([num_total_train_seqs + config.BATCH_SIZE_BIOCHEM, 4*config.MIRLEN, 4*config.SEQLEN])

            # fill features for utr sites for both the guide and passenger strands
            current_ix = 0
            mirlist = train_mirs*config.BATCH_SIZE_REPRESSION
            for mir, (seq_list_guide, seq_list_pass) in zip(mirlist, all_seqs):
                mirseq_one_hot_guide = config.ONE_HOT_DICT[mir]
                mirseq_one_hot_pass = config.ONE_HOT_DICT[mir + '*']

                for seq in seq_list_guide:
                    temp = np.outer(mirseq_one_hot_guide, helpers.one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS))
                    batch_combined_x[current_ix, :, :] = temp
                    current_ix += 1

                for seq in seq_list_pass:
                    temp = np.outer(mirseq_one_hot_pass, helpers.one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS))
                    batch_combined_x[current_ix, :, :] = temp
                    current_ix += 1

            # fill in features for biochem data
            for mir, seq in zip(biochem_train_batch['mir'], biochem_train_batch['seq']):
                mirseq_one_hot = config.ONE_HOT_DICT[mir]
                temp = np.outer(mirseq_one_hot, helpers.one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS))
                batch_combined_x[current_ix, :, :] = temp
                current_ix += 1

            assert(current_ix == batch_combined_x.shape[0])

            batch_combined_x = np.expand_dims((batch_combined_x*4) - 0.25, 3)
            batch_biochem_y = biochem_train_batch[['log ka']].values + config.ZERO_OFFSET # OFFSET

            # run train step
            batch_prefit = prefit.loc[batch_genes]
            # if current_epoch >= config.SWITCH_EPOCH:
            #     data_prefit_xs = np.zeros([config.BATCH_SIZE_REPRESSION, 0])
            #     data_prefit_ys = np.zeros([config.BATCH_SIZE_REPRESSION, 0])
            # else:
            #     x_cols = [x for x in prefit.columns if 'x_' in x]
            #     y_cols = [x for x in prefit.columns if 'y_' in x]
            #     data_prefit_xs = batch_prefit[x_cols].values
            #     data_prefit_ys = batch_prefit[y_cols].values


            # make feed dict for training
            feed_dict = {
                    _keep_prob: config.KEEP_PROB_TRAIN,
                    _phase_train: True,
                    _repression_weight: config.REPRESSION_WEIGHT,
                    _combined_x: batch_combined_x,
                    _biochem_y: batch_biochem_y,
                    _repression_max_size: max_sites,
                    _repression_split_sizes: train_sizes,
                    _repression_y: batch_repression_y,
                    _utr_len: batch_prefit[['utr_length']].values,
                    _let7_sites: let7_sites.loc[batch_genes].values.reshape([config.BATCH_SIZE_REPRESSION,1,let7_num_kds]),
                    _let7_mask: let7_mask.loc[batch_genes].values.reshape([config.BATCH_SIZE_REPRESSION,1,let7_num_kds]),
                    # _prefit_xs: data_prefit_xs,
                    # _prefit_ys: data_prefit_ys
                }
            # print(batch_genes)
            # print(let7_sites.loc[batch_genes].values.reshape([config.BATCH_SIZE_REPRESSION,1,let7_num_kds]))
            # print(let7_mask.loc[batch_genes].values.reshape([config.BATCH_SIZE_REPRESSION,1,let7_num_kds]))
            # print(prefit.loc[batch_genes][['utr_length']].values)
            # print(data_prefit_xs, data_prefit_ys)

            times.append(time.time() - t0)
            t0 = time.time()

            # _ = sess.run(_train_step, feed_dict=feed_dict)
            _, l1, l2, l3, train_loss = sess.run([_train_step, _biochem_loss, _repression_loss,
                                                  _weight_regularize, _loss], feed_dict=feed_dict)

            step += 1

            times2.append(time.time() - t0)

            if (step % config.REPORT_INT) == 0:
            # if next_epoch:

                print(np.mean(times), np.mean(times2))

                # step_list.append(current_epoch)
                step_list.append(step)

                # save model
                # saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=current_epoch)
                
                # calculate and plot train performance
                train_losses.append(train_loss)
                print('Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(current_epoch, train_loss, l1, l2, l3))

                fig = plt.figure(figsize=(7,5))
                plt.plot(step_list, train_losses)
                plt.savefig(os.path.join(options.LOGDIR, 'train_losses.png'))
                plt.close()

                feed_dict = {
                    _keep_prob: 1.0,
                    _phase_train: False,
                    _repression_weight: config.REPRESSION_WEIGHT,
                    _combined_x: batch_combined_x,
                    _biochem_y: batch_biochem_y,
                    _repression_max_size: max_sites,
                    _repression_split_sizes: train_sizes,
                    _repression_y: batch_repression_y,
                    _utr_len: batch_prefit[['utr_length']].values,
                    _let7_sites: let7_sites.loc[batch_genes].values.reshape([config.BATCH_SIZE_REPRESSION,1,let7_num_kds]),
                    _let7_mask: let7_mask.loc[batch_genes].values.reshape([config.BATCH_SIZE_REPRESSION,1,let7_num_kds]),
                    # _prefit_xs: data_prefit_xs,
                    # _prefit_ys: data_prefit_ys
                }

                train_biochem_preds = sess.run(_pred_biochem, feed_dict=feed_dict)
                train_repression_kds = sess.run(_pred_repression_flat, feed_dict=feed_dict)

                fig = plt.figure(figsize=(7,7))
                plt.scatter(train_biochem_preds.flatten(), batch_biochem_y.flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'train_biochem_scatter.png'))
                plt.close()

                fig = plt.figure(figsize=(7,7))
                plt.hist(train_repression_kds.flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'train_repression_kds_hist.png'))
                plt.close()

                train_repression_preds, train_repression_ys = sess.run([_pred_logfc_normed, _repression_y_normed],
                                                                        feed_dict=feed_dict)

                print(train_repression_preds.shape)

                fig = plt.figure(figsize=(7,7))
                for i in range(config.BATCH_SIZE_REPRESSION):
                    blah = pd.DataFrame({'pred': train_repression_preds[i,:], 'actual': train_repression_ys[i,:]})
                    blah = blah.sort_values('pred')
                    plt.plot(blah['pred'], blah['actual'])

                plt.savefig(os.path.join(options.LOGDIR, 'train_repression_fits.png'))
                plt.close()

                fig = plt.figure(figsize=(7,7))
                plt.scatter(train_repression_preds, train_repression_ys)
                plt.savefig(os.path.join(options.LOGDIR, 'train_repression_scatter.png'))
                plt.close()

                # plot weights
                conv_weights = sess.run(_w1)
                xlabels = ['U','A','G','C']
                ylabels = ['A','U','C','G']
                helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1.pdf'))

                # conv_weights = np.abs(sess.run(_w2))
                # conv_weights = np.sum(conv_weights, axis=(2,3))
                # vmin, vmax = np.min(conv_weights), np.max(conv_weights)
                # xlabels = ['s1', 's2']
                # ylabels = ['m2', 'm1']
                # fig = plt.figure(figsize=(4,4))
                # sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                #             cmap=plt.cm.bwr, vmin=vmin, vmax=vmax)
                # plt.savefig(os.path.join(options.LOGDIR, 'convolution2.pdf'))
                # plt.close()

                conv_weights = np.abs(sess.run(_w3))
                conv_weights = np.sum(conv_weights, axis=(2,3))
                vmin, vmax = np.min(conv_weights), np.max(conv_weights)
                xlabels = ['s{}'.format(i+1) for i in range(config.SEQLEN)]
                ylabels = ['m{}'.format(i+1) for i in list(range(config.MIRLEN))[::-1]]
                fig = plt.figure(figsize=(4,4))
                sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                            cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
                plt.savefig(os.path.join(options.LOGDIR, 'convolution3.pdf'))
                plt.close()

                current_freeAGO = sess.run(_freeAGO_init)#.reshape([NUM_TRAIN, 2])
                current_decay = sess.run(_decay)
                current_utr_coef = sess.run(_utr_coef)


                feed_dict = {
                    _keep_prob: 1.0,
                    _phase_train: False,
                    _combined_x: test_kds_combined_x
                }

                pred_test_kds = sess.run(_pred_ind_values, feed_dict=feed_dict)

                fig = plt.figure(figsize=(7,7))
                plt.scatter(pred_test_kds, test_kds_labels, s=20)
                plt.savefig(os.path.join(options.LOGDIR, 'test_kd_scatter.png'))
                plt.close()

                print(current_freeAGO)
                # print(current_decay)
                # print(current_utr_coef)


                if current_epoch == 50:
                    break

            #     # predict nbound for test miRNA
            #     feed_dict = {
            #                     _keep_prob: 1.0,
            #                     _phase_train: False,
            #                     _combined_x: test_combined_x
            #                 }

            #     pred_ind_values_test = sess.run(_pred_ind_values, feed_dict=feed_dict).flatten()
            #     nbound_guide, nbound_pass = [], []
            #     prev = 0
            #     guide = True
            #     for size in test_seq_sizes:
            #         temp = pred_ind_values_test[prev: prev+size]
            #         prev += size
            #         if guide:
            #             ind_nbound = helpers.sigmoid(temp + current_freeAGO)
            #             nbound_guide.append(np.sum(ind_nbound))
            #             guide = False
            #         else:
            #             ind_nbound = helpers.sigmoid(temp + (current_freeAGO - test_freeAGO_diff))
            #             nbound_pass.append(np.sum(ind_nbound))
            #             guide = True

            #     pred_nbound_test = np.array(nbound_guide) + np.array(nbound_pass)

            #     # pred_logfc_test = (pred_nbound_test) * current_slope
            #     pred_logfc_test = -1 * np.log1p(pred_nbound_test)
            #     baselines = baseline_df.loc[test_tpm.index]['nosite_tpm'].values

            #     test_logfc_labels = test_tpm_labels.flatten() - baselines

            #     # calculate and plot test performance
            #     test_losses.append(np.sum((pred_logfc_test - test_logfc_labels)**2)/len(test_tpm))

            #     fig = plt.figure(figsize=(7,5))
            #     plt.plot(step_list, test_losses)
            #     plt.savefig(os.path.join(options.LOGDIR, 'test_losses.png'))
            #     plt.close()

            #     fig = plt.figure(figsize=(7,7))
            #     plt.scatter(pred_logfc_test, test_logfc_labels, s=30)
            #     rsq = helpers.calc_rsq(pred_logfc_test, test_logfc_labels)
            #     rsq2 = stats.linregress(pred_logfc_test, test_logfc_labels)[2]**2
            #     plt.title('R2 = {:.3}, {:.3}'.format(rsq, rsq2))
            #     plt.savefig(os.path.join(options.LOGDIR, 'test_scatter.png'))
            #     plt.close()

            #     fig = plt.figure(figsize=(7,7))
            #     plt.hist(pred_ind_values_test, bins=100)
            #     plt.savefig(os.path.join(options.LOGDIR, 'test_biochem_hist.png'))
            #     plt.close()

            #     fig = plt.figure(figsize=(7,7))
            #     plt.hist(baseline_df['nosite_tpm'], bins=100)
            #     plt.savefig(os.path.join(options.LOGDIR, 'nosite_tpm_hist.png'))
            #     plt.close()

            #     if last_batch:
            #         print(stats.linregress(pred_nbound_test.flatten(), test_logfc_labels.flatten()))
            #         print('Repression epochs: {}'.format(repression_train_data.num_epochs))
            #         print('Biochem epochs: {}'.format(biochem_train_data.num_epochs))
            #         # print('Global slope: {:.3}'.format(sess.run(_slope)))
            #         trained_freeAGO = sess.run(_freeAGO_all).flatten().reshape([NUM_TRAIN, 2])
            #         print('Fitted free AGO:')
            #         for m, f in zip(train_mirs, trained_freeAGO):
            #             print('{}: {:.3}, {:.3}'.format(m, f[0], f[1]))

            #         freeAGO_df = pd.DataFrame({'mir': train_mirs,
            #                                    'guide': trained_freeAGO[:, 0],
            #                                    'passenger': trained_freeAGO[:, 1]})

            #         freeAGO_df.to_csv(os.path.join(options.LOGDIR, 'freeAGO_final.txt'), sep='\t', index=False)
            #         baseline_df.to_csv(os.path.join(options.LOGDIR, 'final_baselines.txt'), sep='\t')

            #         fig = plt.figure(figsize=(7,7))
            #         plt.scatter(baseline_df['nosite_tpm'], baseline_original['nosite_tpm'])
            #         plt.xlabel('new')
            #         plt.ylabel('original')
            #         plt.savefig(os.path.join(options.LOGDIR, 'nosite_scatter.png'))
            #         plt.close()

            #         print('Global decay rate: {:.3}'.format(sess.run(_decay)))
            #         break
