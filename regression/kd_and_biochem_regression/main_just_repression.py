from optparse import OptionParser
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tensorflow as tf

import config, helpers, objects

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-k", "--kdfile", dest="KD_FILE", help="kd data")
    parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    parser.add_option("-i", "--init_baselines", dest="BASELINE_FILE", help="initial baseline data")
    parser.add_option("-m", "--mirna", dest="TEST_MIRNA", help="testing miRNA")
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("-d", "--do_training", dest="DO_TRAINING", help="toggle training", action="store_true", default=False)
    parser.add_option("-p", "--pretrain", dest="PRETRAIN", help="directory with pretrained weights", default=None)

    (options, args) = parser.parse_args()

    MIRLEN = 12
    SEQLEN = 12
    BATCH_SIZE_BIOCHEM = 50
    BATCH_SIZE_REPRESSION = 5
    KEEP_PROB_TRAIN = 0.5
    STARTING_LEARNING_RATE = 0.002
    LAMBDA = 0.001
    NUM_EPOCHS = 30
    REPORT_INT = 50
    ZERO_OFFSET = 0.0
    NORM_RATIO = 4.0

    HIDDEN1 = 4
    HIDDEN2 = 8
    HIDDEN3 = 16

    PRETRAIN_SAVE_PATH = os.path.join(options.LOGDIR, 'pretrain_saved')
    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    MIR_NTS = np.array(['A','T','C','G'])
    SEQ_NTS = np.array(['T','A','G','C'])

    # make dictionary of reverse miRNA sequences trimmed to MIRLEN
    MIRSEQ_DICT_MIRLEN = {x: y[:MIRLEN][::-1] for (x,y) in config.MIRSEQ_DICT.items()}
    ONE_HOT_DICT = {x: helpers.one_hot_encode_nt_new(y, MIR_NTS) for (x,y) in MIRSEQ_DICT_MIRLEN.items()}

    ### READ EXPRESSION DATA ###
    tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)

    MIRS = [x for x in tpm.columns if ('mir' in x) or ('lsy' in x)]
    assert options.TEST_MIRNA in MIRS

    # split miRNAs into training and testing
    # train_mirs = [m for m in MIRS if m != options.TEST_MIRNA]
    train_mirs = MIRS
    test_mirs = [options.TEST_MIRNA]
    print('Train miRNAs: {}'.format(train_mirs))
    print('Test miRNAs: {}'.format(test_mirs))
    NUM_TRAIN = len(train_mirs)
    NUM_TEST = len(test_mirs)

    # split tpm data into training and testing
    train_tpm = tpm[train_mirs + ['Sequence']]
    test_tpm = tpm[test_mirs + ['Sequence']]

    ### READ KD DATA ###
    data = pd.read_csv(options.KD_FILE, sep='\t')
    data.columns = ['mir','mirseq_full','seq','log kd','stype']

    # print(data['mir'].unique())
    # data = data[data['mir'] != 'mir7']
    # print(data['mir'].unique())

    # zero-center and normalize Ka's
    data['log ka'] = ((-1.0 * data['log kd']) + ZERO_OFFSET) / NORM_RATIO
    data['mirseq'] = [MIRSEQ_DICT_MIRLEN[mir] for mir in data['mir']]
    data['sitem8'] = [helpers.rev_comp(mirseq[1:8]) for mirseq in data['mirseq_full']]
    data['color'] = [helpers.get_color(sitem8, seq) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]
    data['color2'] = [helpers.get_color(sitem8, seq[2:10]) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

    # get rid of sequences with sites out of register
    print(len(data))
    data = data[data['color'] == data['color2']].drop('color2',1)
    print(len(data))

    # create data object
    biochem_train_data = objects.BiochemData(data, cutoff=0.9)
    biochem_train_data.shuffle()

    ### READ INITIAL BASELINE ###
    baseline_df = pd.read_csv(options.BASELINE_FILE, sep='\t', index_col=0)
    assert (len(baseline_df) == len(tpm))
    NUM_GENES = len(baseline_df)
    baseline_df = baseline_df.loc[tpm.index]
    # gene_order = tpm.index

    # make data objects for repression training data
    repression_train_data = objects.RepressionDataNew_with_passenger(train_tpm)
    repression_train_data.shuffle()
    repression_train_data.get_seqs(train_mirs)

    # get test miRNA labels
    test_tpm_labels = test_tpm[test_mirs].values

    # get test sequences for sense strand
    test_mirseq = MIRSEQ_DICT_MIRLEN[options.TEST_MIRNA]
    test_seqs = []
    test_site = config.SITE_DICT[options.TEST_MIRNA]
    num_total_test_seqs = 0
    for utr in test_tpm['Sequence']:
        seqs = helpers.get_seqs(utr, test_site, only_canon=False)
        test_seqs.append(seqs)
        num_total_test_seqs += len(seqs)

    test_combined_x = np.zeros([num_total_test_seqs, 4*MIRLEN, 4*SEQLEN])
    test_seq_utr_boundaries = [0]
    current_ix = 0
    for seq_list in test_seqs:

        if len(seq_list) == 0:
            test_seq_utr_boundaries.append(current_ix)

        else:
            for seq in seq_list:
                test_combined_x[current_ix, :, :] = helpers.make_square(test_mirseq, seq)
                current_ix += 1

            test_seq_utr_boundaries.append(current_ix)
    
    test_combined_x = np.expand_dims(test_combined_x, 3)

    # get test sequences for passenger strand
    test_mirseq = MIRSEQ_DICT_MIRLEN[options.TEST_MIRNA + '*']
    test_seqs = []
    test_site = config.SITE_DICT[options.TEST_MIRNA + '*']
    num_total_test_seqs = 0
    for utr in test_tpm['Sequence']:
        seqs = helpers.get_seqs(utr, test_site, only_canon=False)
        test_seqs.append(seqs)
        num_total_test_seqs += len(seqs)

    test_combined_x_pass = np.zeros([num_total_test_seqs, 4*MIRLEN, 4*SEQLEN])
    test_seq_utr_boundaries_pass = [0]
    current_ix = 0
    for seq_list in test_seqs:

        if len(seq_list) == 0:
            test_seq_utr_boundaries_pass.append(current_ix)

        else:
            for seq in seq_list:
                test_combined_x_pass[current_ix, :, :] = helpers.make_square(test_mirseq, seq)
                current_ix += 1

            test_seq_utr_boundaries_pass.append(current_ix)
    
    test_combined_x_pass = np.expand_dims(test_combined_x_pass, 3)

    test_freeAGO_site = config.FREEAGO_SITE_DICT[options.TEST_MIRNA]
    test_freeAGO_pass = config.FREEAGO_PASS_DICT[options.TEST_MIRNA]
    test_freeAGO_diff = test_freeAGO_site - test_freeAGO_pass
    

    ### DEFINE MODEL ###

    # reset and build the neural network
    tf.reset_default_graph()

    # start session
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:

        # create placeholders for input data
        _keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        _phase_train = tf.placeholder(tf.bool, name='phase_train')
        _combined_x = tf.placeholder(tf.float32, shape=[None, 4 * MIRLEN, 4 * SEQLEN, 1], name='biochem_x')
        _pretrain_y =  tf.placeholder(tf.float32, shape=[None, 1], name='pretrain_y')

        # add layer 1
        with tf.name_scope('layer1'):
            _w1, _b1 = helpers.get_conv_params(4, 4, 1, HIDDEN1, 'layer1')
            _preactivate1 = tf.nn.conv2d(_combined_x, _w1, strides=[1, 4, 4, 1], padding='VALID') + _b1

            _preactivate1_bn = tf.layers.batch_normalization(_preactivate1, axis=1, training=_phase_train)

            _layer1 = tf.nn.relu(_preactivate1_bn)

        # add layer 2
        with tf.name_scope('layer2'):
            _w2, _b2 = helpers.get_conv_params(2, 2, HIDDEN1, HIDDEN2, 'layer2')
            _preactivate2 = tf.nn.conv2d(_layer1, _w2, strides=[1, 1, 1, 1], padding='SAME') + _b2

            _preactivate2_bn = tf.layers.batch_normalization(_preactivate2, axis=1, training=_phase_train)

            _layer2 = tf.nn.relu(_preactivate2_bn)

        # add layer 3
        with tf.name_scope('layer3'):
            _w3, _b3 = helpers.get_conv_params(MIRLEN, SEQLEN, HIDDEN2, HIDDEN3, 'layer3')
            _preactivate3 = tf.nn.conv2d(_layer2, _w3, strides=[1, MIRLEN, SEQLEN, 1], padding='VALID') + _b3

            _preactivate3_bn = tf.layers.batch_normalization(_preactivate3, axis=1, training=_phase_train)

            _layer3 = tf.nn.relu(_preactivate3_bn)

        # add dropout
        with tf.name_scope('dropout'):
            _dropout = tf.nn.dropout(_layer3, _keep_prob)

        # reshape to 1D tensor
        _layer_flat = tf.reshape(_dropout, [-1, HIDDEN3])

        # add last layer
        with tf.name_scope('final_layer'):
            with tf.name_scope('weights'):
                _w4 = tf.get_variable("final_layer_weight", shape=[HIDDEN3, 1],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                tf.add_to_collection('weight', _w4)
            with tf.name_scope('biases'):
                _b4 = tf.get_variable("final_layer_bias", shape=[1],
                                    initializer=tf.constant_initializer(0.0))
                tf.add_to_collection('bias', _b4)

            # apply final layer
            _pred_ind_values = tf.matmul(_layer_flat, _w4) + _b4

        _pretrain_loss = tf.nn.l2_loss(tf.subtract(_pred_ind_values, _pretrain_y))
        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops):
            _train_step_pretrain = tf.train.AdamOptimizer(0.01).minimize(_pretrain_loss)

        saver_pretrain = tf.train.Saver()

        if options.DO_TRAINING:

            # make variables for real training
            _repression_mask_site = tf.placeholder(tf.float32, shape=[None, None, None], name='repression_mask_site')
            _repression_mask_pass = tf.placeholder(tf.float32, shape=[None, None, None], name='repression_mask_pass')
            _repression_y = tf.placeholder(tf.float32, shape=[None, None], name='repression_y')
        
            _freeAGO_site = tf.get_variable('freeAGO_site', shape=[1,NUM_TRAIN,1],
                                            initializer=tf.constant_initializer(-5.0 - ZERO_OFFSET))
            _freeAGO_pass = tf.get_variable('freeAGO_pass', shape=[1,NUM_TRAIN,1],
                                            initializer=tf.constant_initializer(-7.0 - ZERO_OFFSET))
            _slope = tf.get_variable('slope', shape=(), initializer=tf.constant_initializer(-0.51023716))

            _pred_repression = tf.reshape(_pred_ind_values, [BATCH_SIZE_REPRESSION*2, NUM_TRAIN, -1])

            _pred_repression_site = _pred_repression[:BATCH_SIZE_REPRESSION, :, :] * NORM_RATIO
            _pred_repression_pass = _pred_repression[BATCH_SIZE_REPRESSION:, :, :] * NORM_RATIO 

            # calculate predicted number bound and predicted log fold-change
            _pred_nbound_site = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO_site + _pred_repression_site), _repression_mask_site), axis=2)
            _pred_nbound_pass = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO_pass + _pred_repression_pass), _repression_mask_pass), axis=2)
            _pred_logfc = (_pred_nbound_site + _pred_nbound_pass) * _slope

            _weight_regularize = tf.multiply(tf.nn.l2_loss(_w1) \
                                    + tf.nn.l2_loss(_w2) \
                                    + tf.nn.l2_loss(_w3) \
                                    + tf.nn.l2_loss(_w4), LAMBDA)

            _intercept = tf.reshape((tf.reduce_mean(_repression_y, axis=1) - tf.reduce_mean(_pred_logfc, axis=1)), [-1, 1])
            _pred_tpm = _pred_logfc + _intercept

            _repression_loss = _repression_weight * tf.nn.l2_loss(tf.subtract(_pred_tpm, _repression_y)) / NUM_TRAIN

            _loss = _repression_loss + _weight_regularize

            _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(_update_ops):
                _train_step = tf.train.AdamOptimizer(STARTING_LEARNING_RATE).minimize(_loss)

            saver = tf.train.Saver(max_to_keep=NUM_EPOCHS)


        ### PRETRAIN MODEL ###
    
        sess.run(tf.global_variables_initializer())

        if options.PRETRAIN is not None:

            if options.PRETRAIN  == 'pretrain':

                print("Doing pre-training")

                # plot random initialized weights
                conv_weights = sess.run(_w1)
                xlabels = ['U','A','G','C']
                ylabels = ['A','U','C','G']
                helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1_start.pdf'))

                # pretrain on generated site-type-based data
                losses = []
                for pretrain_step in range(2000):
                    pretrain_batch_x, pretrain_batch_y = helpers.make_pretrain_data(100, MIRLEN, SEQLEN)
                    pretrain_batch_y = (pretrain_batch_y + ZERO_OFFSET) / NORM_RATIO

                    feed_dict = {
                                    _keep_prob: KEEP_PROB_TRAIN,
                                    _phase_train: True,
                                    _repression_weight: REPRESSION_WEIGHT,
                                    _combined_x: pretrain_batch_x,
                                    _pretrain_y: pretrain_batch_y
                                }

                    _, l = sess.run([_train_step_pretrain, _pretrain_loss], feed_dict=feed_dict)
                    losses.append(l)

                    if (pretrain_step % 100) == 0:
                        print(pretrain_step)

                train_pred = sess.run(_pred_ind_values, feed_dict=feed_dict)

                fig = plt.figure(figsize=(7,7))
                plt.scatter(train_pred.flatten(), pretrain_batch_y.flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'pretrain_train_scatter.png'))
                plt.close()

                test_x, test_y = helpers.make_pretrain_data(100, MIRLEN, SEQLEN)
                test_y = (test_y + ZERO_OFFSET) / NORM_RATIO
                feed_dict = {
                                _keep_prob: 1.0,
                                _phase_train: False,
                                _combined_x: test_x
                            }
                pred_pretrain = sess.run(_pred_ind_values, feed_dict=feed_dict)


                fig = plt.figure(figsize=(7,7))
                plt.scatter(train_pred.flatten(), pretrain_batch_y.flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'pretrain_train_scatter.png'))
                plt.close()

                fig = plt.figure(figsize=(7,7))
                plt.scatter(pred_pretrain.flatten(), test_y.flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'pretrain_test_scatter.png'))
                plt.close()

                fig = plt.figure(figsize=(7,5))
                plt.plot(losses)
                plt.savefig(os.path.join(options.LOGDIR, 'pretrain_losses.png'))
                plt.close()

                saver.save(sess, os.path.join(PRETRAIN_SAVE_PATH, 'model'))

                print("Finished pre-training")

            else:
                # restore previously pretrained weights
                latest = tf.train.latest_checkpoint(options.PRETRAIN)
                print('Restoring from {}'.format(latest))
                saver_pretrain.restore(sess, latest)

            # plot weights after pre-training
            conv_weights = sess.run(_w1)
            xlabels = ['U','A','G','C']
            ylabels = ['A','U','C','G']
            helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1_pretrained.pdf'))

            conv_weights = np.abs(sess.run(_w3))
            conv_weights = np.sum(conv_weights, axis=(2,3))
            vmin, vmax = np.min(conv_weights), np.max(conv_weights)
            xlabels = ['s{}'.format(i+1) for i in range(SEQLEN)]
            ylabels = ['m{}'.format(i+1) for i in list(range(MIRLEN))[::-1]]
            fig = plt.figure(figsize=(4,4))
            sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                        cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
            plt.savefig(os.path.join(options.LOGDIR, 'convolution3_pretrained.pdf'))
            plt.close()

        ### TRAIN MODEL ###

        if options.DO_TRAINING:

            print("Training now on {}".format(options.TEST_MIRNA))

            # reset later variables
            sess.run(_w3.initializer)
            sess.run(_b3.initializer)
            sess.run(_w4.initializer)
            sess.run(_b4.initializer)

            conv_weights = np.abs(sess.run(_w3))
            conv_weights = np.sum(conv_weights, axis=(2,3))
            vmin, vmax = np.min(conv_weights), np.max(conv_weights)
            xlabels = ['s{}'.format(i+1) for i in range(SEQLEN)]
            ylabels = ['m{}'.format(i+1) for i in list(range(MIRLEN))[::-1]]
            fig = plt.figure(figsize=(4,4))
            sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                        cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
            plt.savefig(os.path.join(options.LOGDIR, 'convolution3_reset.pdf'))
            plt.close()

            step_list = []
            train_losses = []
            test_losses = []
            last_batch = False

            step = -1
            current_epoch = 0
            while True:

                # get repression data batch
                batch_genes, next_epoch, all_seqs_site, all_seqs_pass, num_sites, batch_repression_y = repression_train_data.get_next_batch(BATCH_SIZE_REPRESSION, train_mirs)

                if next_epoch:
                    current_epoch += 1
                    # REPRESSION_WEIGHT += 0.2
                    if repression_train_data.num_epochs >= NUM_EPOCHS:
                        last_batch = True

                # if none of the genes have sites, continue
                if num_sites == 0:
                    continue

                # get biochem data batch
                _, biochem_train_batch = biochem_train_data.get_next_batch(BATCH_SIZE_BIOCHEM)

                # fill feature arrays
                batch_combined_x = np.zeros([(BATCH_SIZE_REPRESSION * 2 * NUM_TRAIN * num_sites), 4*MIRLEN, 4*SEQLEN])
                batch_repression_mask_site = np.zeros([BATCH_SIZE_REPRESSION, NUM_TRAIN, num_sites])
                batch_repression_mask_pass = np.zeros([BATCH_SIZE_REPRESSION, NUM_TRAIN, num_sites])

                # put in site sequences
                for counter1, big_seq_list in enumerate(all_seqs_site):

                    for counter2, (mir, seq_list) in enumerate(zip(train_mirs, big_seq_list)):

                        if len(seq_list) == 0:
                            continue

                        mirseq_one_hot = ONE_HOT_DICT[mir]
                        current = (counter1 * NUM_TRAIN * num_sites) + (counter2 * num_sites)
                        for seq in seq_list:
                            temp = np.outer(mirseq_one_hot, helpers.one_hot_encode_nt_new(seq, SEQ_NTS))
                            batch_combined_x[current, :, :] = temp - 0.25
                            current += 1
                        batch_repression_mask_site[counter1, counter2, :len(seq_list)] = 1.0

                # put in passenger strand sequences
                for counter1, big_seq_list in enumerate(all_seqs_pass):

                    for counter2, (mir, seq_list) in enumerate(zip(train_mirs, big_seq_list)):

                        if len(seq_list) == 0:
                            continue

                        mirseq_one_hot = ONE_HOT_DICT[mir + '*']
                        current = ((counter1 + BATCH_SIZE_REPRESSION) * NUM_TRAIN * num_sites) + (counter2 * num_sites)
                        for seq in seq_list:
                            temp = np.outer(mirseq_one_hot, helpers.one_hot_encode_nt_new(seq, SEQ_NTS))
                            batch_combined_x[current, :, :] = temp - 0.25
                            current += 1
                        batch_repression_mask_pass[counter1, counter2, :len(seq_list)] = 1.0


                batch_combined_x = np.expand_dims(batch_combined_x, 3)
                batch_biochem_y = biochem_train_batch[['log ka']].values

                # make feed dict for training
                feed_dict = {
                        _keep_prob: KEEP_PROB_TRAIN,
                        _phase_train: True,
                        _combined_x: batch_combined_x,
                        _repression_mask_site: batch_repression_mask_site,
                        _repression_mask_pass: batch_repression_mask_pass,
                        _repression_y: batch_repression_y
                    }

                # run train step
                _, batch_intercept = sess.run([_train_step, _intercept], feed_dict=feed_dict)

                # update gene baseline expression
                baseline_df.loc[batch_genes, 'nosite_tpm'] = batch_intercept.flatten()


                # _, l1, l2, l3 = sess.run([_train_step, _biochem_loss, _loss, _weight_regularize], feed_dict=feed_dict)
                step += 1

                if (step % REPORT_INT) == 0:
                # if next_epoch:

                    # save model
                    saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=step)
                    # saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=current_epoch)

                    train_loss = sess.run(_loss, feed_dict=feed_dict)
                    # print(l1, l2, l3)
                    
                    step_list.append(step)
                    # step_list.append(current_epoch)
                    
                    train_losses.append(train_loss)

                    feed_dict = {
                        _keep_prob: 1.0,
                        _phase_train: False,
                        _combined_x: batch_combined_x,
                        _repression_mask_site: batch_repression_mask_site,
                        _repression_mask_pass: batch_repression_mask_pass,
                        _repression_y: batch_repression_y
                    }

                    train_biochem_preds, intercept = sess.run([_pred_ind_values, _intercept], feed_dict=feed_dict)

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(train_biochem_preds.flatten(), batch_biochem_y.flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'test_biochem_scatter.png'))
                    plt.close()

                    train_repression_preds = sess.run(_pred_logfc, feed_dict=feed_dict)

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(train_repression_preds, batch_repression_y - intercept)
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
                    xlabels = ['s{}'.format(i+1) for i in range(SEQLEN)]
                    ylabels = ['m{}'.format(i+1) for i in list(range(MIRLEN))[::-1]]
                    fig = plt.figure(figsize=(4,4))
                    sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                                cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
                    plt.savefig(os.path.join(options.LOGDIR, 'convolution3.pdf'))
                    plt.close()

                    current_freeAGO = np.mean(sess.run(_freeAGO_site))
                    current_slope = sess.run(_slope)
                    # print('current free AGO: {:.3}'.format(current_freeAGO))
                    # print('current slope: {:.3}'.format(current_slope))

                    # print(current_freeAGO, current_freeAGO - test_freeAGO_diff)

                    # get nbound for miRNA strand
                    feed_dict = {
                                    _keep_prob: 1.0,
                                    _phase_train: False,
                                    _combined_x: test_combined_x
                                }

                    pred_ind_values_test = sess.run(_pred_ind_values, feed_dict=feed_dict)
                    pred_ind_nbound_test = 1.0 / (1.0 + np.exp((-1.0 * pred_ind_values_test.flatten() * NORM_RATIO) - current_freeAGO))
                    pred_nbound_test = []
                    prev = test_seq_utr_boundaries[0]
                    for bound in test_seq_utr_boundaries[1:]:
                        pred_nbound_test.append(np.sum(pred_ind_nbound_test[prev:bound]))
                        prev = bound

                    pred_nbound_test_site = np.array(pred_nbound_test)

                    # get nbound for passenger strand
                    feed_dict = {
                                    _keep_prob: 1.0,
                                    _phase_train: False,
                                    _combined_x: test_combined_x_pass
                                }

                    pred_ind_values_test = sess.run(_pred_ind_values, feed_dict=feed_dict)
                    pred_ind_nbound_test = 1.0 / (1.0 + np.exp((-1.0 * pred_ind_values_test.flatten() * NORM_RATIO) - (current_freeAGO - test_freeAGO_diff)))
                    pred_nbound_test = []
                    prev = test_seq_utr_boundaries[0]
                    for bound in test_seq_utr_boundaries[1:]:
                        pred_nbound_test.append(np.sum(pred_ind_nbound_test[prev:bound]))
                        prev = bound

                    pred_nbound_test_pass = np.array(pred_nbound_test)

                    pred_nbound_test = pred_nbound_test_site + pred_nbound_test_pass


                    pred_logfc_test = (pred_nbound_test) * current_slope

                    test_logfc_labels = test_tpm_labels - baseline_df.loc[test_tpm.index][['nosite_tpm']].values

                    test_losses.append(np.sum((pred_logfc_test - test_logfc_labels.flatten())**2)/len(test_logfc_labels))

                    fig = plt.figure(figsize=(7,5))
                    plt.plot(step_list, test_losses)
                    plt.savefig(os.path.join(options.LOGDIR, 'test_losses.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7,5))
                    plt.plot(step_list, np.log(np.array(train_losses)))
                    plt.savefig(os.path.join(options.LOGDIR, 'train_losses_log.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(pred_logfc_test, test_logfc_labels.flatten(), s=30)
                    rsq = helpers.calc_rsq(pred_logfc_test, test_logfc_labels.flatten())
                    rsq2 = stats.linregress(pred_nbound_test, test_logfc_labels.flatten())[2]**2
                    plt.title('R2 = {:.3}, {:.3}'.format(rsq, rsq2))
                    plt.savefig(os.path.join(options.LOGDIR, 'test_scatter.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7,7))
                    plt.hist(pred_ind_values_test, bins=100)
                    plt.savefig(os.path.join(options.LOGDIR, 'test_biochem_hist.png'))
                    plt.close()

                    if last_batch:
                        print(stats.linregress(pred_nbound_test.flatten(), test_logfc_labels.flatten()))
                        print('Repression epochs: {}'.format(repression_train_data.num_epochs))
                        print('Biochem epochs: {}'.format(biochem_train_data.num_epochs))
                        print('Global slope: {:.3}'.format(sess.run(_slope)))
                        trained_freeAGO_site = sess.run(_freeAGO_site).flatten()
                        trained_freeAGO_pass = sess.run(_freeAGO_pass).flatten()
                        print('Fitted free AGO:')
                        for m, f1, f2 in zip(train_mirs, trained_freeAGO_site, trained_freeAGO_pass):
                            print('{}: {:.3}, {:.3}'.format(m, f1, f2))

                        baseline_df.to_csv(os.path.join(options.LOGDIR, 'final_baselines.txt'), sep='\t')
                        break
