from optparse import OptionParser
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
# import seaborn as sns
import tensorflow as tf

import helpers
import objects

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
    BATCH_SIZE_BIOCHEM = 100
    BATCH_SIZE_REPRESSION = 5
    KEEP_PROB_TRAIN = 0.5
    STARTING_LEARNING_RATE = 0.001
    LAMBDA = 0.00
    NUM_EPOCHS = 2
    REPORT_INT = 100
    TPM_WEIGHT = 1

    HIDDEN1 = 4
    HIDDEN2 = 8
    HIDDEN3 = 32

    PRETRAIN_SAVE_PATH = os.path.join(options.LOGDIR, 'pretrain_saved')
    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)


    MIRSEQ_DICT_MIRLEN = {x: y[:MIRLEN][::-1] for (x,y) in helpers.MIRSEQ_DICT.items()}

    # metafile = open(os.path.join(options.LOGDIR, 'params.txt'), 'w')
    # for key in sorted(params.keys()):
    #     metafile.write('{}: {}\n'.format(key, params[key]))

    ### READ EXPRESSION DATA ###
    tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)
    print(tpm.head())

    MIRS = [x for x in tpm.columns if ('mir' in x) or ('lsy' in x)]

    assert options.TEST_MIRNA in MIRS

    # split miRNAs into training and testing
    train_mirs = [m for m in MIRS if m != options.TEST_MIRNA]
    # train_mirs = MIRS
    # train_mirs = ['mir1','mir124']
    test_mirs = [options.TEST_MIRNA]
    print(train_mirs)
    print(test_mirs)
    NUM_TRAIN = len(train_mirs)
    NUM_TEST = len(test_mirs)

    # split tpm data into training and testing
    train_tpm = tpm[train_mirs + ['Sequence']]
    test_tpm = tpm[test_mirs + ['Sequence']]

    ### READ KD DATA ###
    data = pd.read_csv(options.KD_FILE, sep='\t')
    data.columns = ['mir','mirseq_full','seq','log kd','stype']
    print(data['stype'].unique())
    data['log ka'] = (-1 * data['log kd'])
    data['mirseq'] = [MIRSEQ_DICT_MIRLEN[mir] for mir in data['mir']]
    data['sitem8'] = [helpers.rev_comp(mirseq[1:8]) for mirseq in data['mirseq_full']]
    data['color'] = [helpers.get_color_old(sitem8, seq) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]
    data['color2'] = [helpers.get_color_old(sitem8, seq[2:10]) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

    # get rid of sequences with sites out of register
    print(len(data))
    data = data[data['color'] == data['color2']].drop('color2',1)
    print(len(data))

    print(data['stype'].unique())

    # create data object
    biochem_train_data = objects.BiochemData(data, cutoff=0.9)
    biochem_train_data.shuffle()

    ### READ INITIAL BASELINE ###
    baseline_init = pd.read_csv(options.BASELINE_FILE, sep='\t', index_col=0)
    assert (len(baseline_init) == len(tpm))
    NUM_GENES = len(baseline_init)
    baseline_init = baseline_init.loc[tpm.index]['nosite_tpm'].values.reshape([NUM_GENES, 1])

    train_tpm[train_mirs] = train_tpm[train_mirs].values - baseline_init
    test_tpm[test_mirs] = test_tpm[test_mirs].values - baseline_init

    # make data objects for repression training data
    repression_train_data = objects.RepressionData(train_tpm)
    repression_train_data.shuffle()

    # test on a subset of the test data to speed up testing
    subset = np.random.choice(np.arange(len(test_tpm)), size=500)
    test_tpm = test_tpm.iloc[subset]
    test_logfc_labels = test_tpm[test_mirs].values

    test_mirseq = MIRSEQ_DICT_MIRLEN[options.TEST_MIRNA]
    test_seqs = []
    for utr in test_tpm['Sequence']:
        _, seqs = helpers.get_tpm_seqs(utr, [options.TEST_MIRNA])
        test_seqs.append(seqs[0])
    

    ### DEFINE MODEL ###

    # reset and build the neural network
    tf.reset_default_graph()

    # start session
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:

        _keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        _phase_train = tf.placeholder(tf.bool, name='phase_train')

        _kd_x = tf.placeholder(tf.float32, shape=[None, 4 * MIRLEN, 4 * SEQLEN, 1], name='kd_x')
        _kd_y = tf.placeholder(tf.float32, shape=[None, 1], name='kd_y')
        _kd_mask = tf.placeholder(tf.float32, shape=[None, 1], name='kd_mask')
        _tpm_mask = tf.placeholder(tf.float32, shape=[None, None, None], name='tpm_mask')
        _tpm_y = tf.placeholder(tf.float32, shape=[None, None], name='tpm_y')
        _pretrain_y =  tf.placeholder(tf.float32, shape=[None, 1], name='pretrain_y')

        # _freeAGO = tf.get_variable('freeAGO', shape=[1,NUM_TRAIN,1], initializer=tf.constant_initializer(0.0))
        _freeAGO = tf.get_variable('freeAGO', shape=[1,NUM_TRAIN,1], initializer=tf.constant_initializer(-5.0))
        _slope = tf.get_variable('slope', shape=(), initializer=tf.constant_initializer(-0.51023716), trainable=False)
        # intercept = tf.get_variable('intercept', shape=[NUM_GENES],
        #                             initializer=tf.constant_initializer(baseline_init), trainable=False)

        with tf.name_scope('layer1'):
            _w1, _b1 = helpers.get_conv_params(4, 4, 1, HIDDEN1, 'layer1')
            _preactivate1_kd = tf.nn.conv2d(_kd_x, _w1, strides=[1, 4, 4, 1], padding='VALID') + _b1

            _preactivate1_kd_bn = tf.layers.batch_normalization(_preactivate1_kd, axis=1, training=_phase_train)

            _layer1_kd = tf.nn.relu(_preactivate1_kd_bn)

        with tf.name_scope('layer2'):
            _w2, _b2 = helpers.get_conv_params(2, 2, HIDDEN1, HIDDEN2, 'layer2')
            _preactivate2_kd = tf.nn.conv2d(_layer1_kd, _w2, strides=[1, 1, 1, 1], padding='SAME') + _b2

            _preactivate2_kd_bn = tf.layers.batch_normalization(_preactivate2_kd, axis=1, training=_phase_train)

            _layer2_kd = tf.nn.relu(_preactivate2_kd_bn)

        with tf.name_scope('layer3'):
            _w3, _b3 = helpers.get_conv_params(MIRLEN, SEQLEN, HIDDEN2, HIDDEN3, 'layer3')
            _preactivate3_kd = tf.nn.conv2d(_layer2_kd, _w3, strides=[1, MIRLEN, SEQLEN, 1], padding='VALID') + _b3

            _preactivate3_kd_bn = tf.layers.batch_normalization(_preactivate3_kd, axis=1, training=_phase_train)

            _layer3_kd = tf.nn.relu(_preactivate3_kd_bn)

        # add dropout
        with tf.name_scope('dropout'):
            _dropout_kd = tf.nn.dropout(_layer3_kd, _keep_prob)

        # reshape to 1D tensor
        _layer_flat_kd = tf.reshape(_dropout_kd, [-1, HIDDEN3])

        # add last layer
        with tf.name_scope('final_layer'):
            with tf.name_scope('weights'):
                _w4 = tf.get_variable("final_layer_weight", shape=[HIDDEN3, 1],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))

                # add variable to collection of variables
                tf.add_to_collection('weight', _w4)
            with tf.name_scope('biases'):
                _b4 = tf.get_variable("final_layer_bias", shape=[1],
                                    initializer=tf.constant_initializer(0.0))

                # add variable to collection of variables
                tf.add_to_collection('bias', _b4)

            # split into kd outputs and tpm outputs
            _pred_kd_ind = tf.matmul(_layer_flat_kd, _w4) + _b4
            _pred_kd = _pred_kd_ind[-1 * BATCH_SIZE_BIOCHEM:, :1]
            _pred_kd_tpm_flat = _pred_kd_ind[:-1 * BATCH_SIZE_BIOCHEM, :1]
            _pred_kd_tpm = tf.reshape(_pred_kd_tpm_flat, [BATCH_SIZE_REPRESSION, NUM_TRAIN, -1])

        _pred_nbound = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO + _pred_kd_tpm), _tpm_mask), axis=2)
        _pred_tpm = (_pred_nbound * _slope)
            

        _weight_regularize = tf.multiply(tf.nn.l2_loss(_w1) \
                                + tf.nn.l2_loss(_w2) \
                                + tf.nn.l2_loss(_w3) \
                                + tf.nn.l2_loss(_w4), LAMBDA)

        _kd_loss = tf.nn.l2_loss(tf.subtract(_pred_kd, _kd_y)) / BATCH_SIZE_BIOCHEM
        _tpm_loss = TPM_WEIGHT * tf.nn.l2_loss(tf.subtract(_pred_tpm, _tpm_y)) / NUM_TRAIN
        _pretrain_loss = tf.nn.l2_loss(tf.subtract(_pred_kd_ind, _pretrain_y))

        _loss = _kd_loss + _tpm_loss + _weight_regularize
        # _loss = _tpm_loss + _weight_regularize
        # _loss = _kd_loss + _weight_regularize

        # train_step = tf.train.AdamOptimizer(STARTING_LEARNING_RATE).minimize(loss)

        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops):
            _train_step = tf.train.AdamOptimizer(STARTING_LEARNING_RATE).minimize(_loss)
            _train_step_pretrain = tf.train.AdamOptimizer(0.01).minimize(_pretrain_loss)


        merged = tf.summary.merge_all()
        saver = tf.train.Saver()


        ### RUN MODEL ###
        step_list = []
        train_losses = []
        test_losses = []
    
        sess.run(tf.global_variables_initializer())

        if options.PRETRAIN is None:

            print("Doing pre-training")

            conv_weights = sess.run(_w1)
            xlabels = ['U','A','G','C']
            ylabels = ['A','U','C','G']
            helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1_start.pdf'))

            losses = []
            for pretrain_step in range(1000):
                pretrain_batch_x, pretrain_batch_y = helpers.make_pretrain_data(100, MIRLEN, SEQLEN)

                feed_dict = {
                                _keep_prob: KEEP_PROB_TRAIN,
                                _phase_train: True,
                                _kd_x: pretrain_batch_x,
                                _pretrain_y: pretrain_batch_y
                            }

                _, l = sess.run([_train_step_pretrain, _pretrain_loss], feed_dict=feed_dict)
                losses.append(l)

            train_pred = sess.run(_pred_kd_ind, feed_dict=feed_dict)

            fig = plt.figure(figsize=(7,7))
            plt.scatter(train_pred.flatten(), pretrain_batch_y.flatten())
            plt.savefig(os.path.join(options.LOGDIR, 'pretrain_train_scatter.png'))
            plt.close()

            test_x, test_y = helpers.make_pretrain_data(100, MIRLEN, SEQLEN)
            feed_dict = {
                            _keep_prob: 1.0,
                            _phase_train: False,
                            _kd_x: test_x
                        }
            pred_pretrain = sess.run(_pred_kd_ind, feed_dict=feed_dict)


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

            conv_weights = sess.run(_w1)
            xlabels = ['U','A','G','C']
            ylabels = ['A','U','C','G']
            helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1_pretrained.pdf'))

            saver.save(sess, os.path.join(PRETRAIN_SAVE_PATH, 'model'))

            print("Finished pre-training")

        else:
            latest = tf.train.latest_checkpoint(options.PRETRAIN)
            print('Restoring from {}'.format(latest))
            saver.restore(sess, latest)


        if options.DO_TRAINING:
            print("Training now on {}".format(options.TEST_MIRNA))

            step = 0
            while True:

                # get repression data batch
                repression_train_batch = repression_train_data.get_next_batch(BATCH_SIZE_REPRESSION)
                if repression_train_data.num_epochs >= NUM_EPOCHS:
                    break

                all_seqs = []
                num_sites = 0
                # num_sites_per_gene = []
                for tpm_row in repression_train_batch.iterrows():
                    utr = tpm_row[1]['Sequence']
                    gene_seqs = []
                    # num_sites_gene = []
                    for mir in train_mirs:
                        seqs = helpers.get_seqs(utr, helpers.SITE_DICT[mir])
                        gene_seqs.append(seqs)
                        len_temp = len(seqs)
                        # num_sites_gene.append(len_temp)

                        if len_temp > num_sites:
                            num_sites = len_temp
                    all_seqs.append(gene_seqs)
                    # num_sites_per_gene.append(num_sites_gene)

                # num_sites_per_gene = np.array(num_sites_per_gene)

                if num_sites == 0:
                    continue

                # get biochem data batch
                biochem_train_batch = biochem_train_data.get_next_batch(BATCH_SIZE_BIOCHEM)

                # ######################
                # t0 = time.time()
                # batch_tpm_kd_x = []
                # batch_tpm_mask = []
                # for big_seq_list in all_seqs:

                #     big_mask_temp = []
                #     for mir, seq_list in zip(train_mirs, big_seq_list):
                #         mirseq = MIRSEQ_DICT_MIRLEN[mir]
                #         batch_tpm_kd_x += [helpers.make_square(mirseq, seq).tolist() for seq in seq_list]
                #         batch_tpm_kd_x += [np.zeros((4*MIRLEN,4*SEQLEN)).tolist() for _ in range(num_sites - len(seq_list))]

                #         mask_temp = [1.0 for _ in range(len(seq_list))] + [0.0 for _ in range(num_sites - len(seq_list))]
                #         big_mask_temp.append(mask_temp)

                #     batch_tpm_mask.append(big_mask_temp)

                # batch_tpm_mask_original = np.array(batch_tpm_mask)
                # batch_tpm_y = repression_train_batch[train_mirs].values

                # for mirseq, seq in zip(biochem_train_batch['mirseq'], biochem_train_batch['seq']):
                #     batch_tpm_kd_x.append(helpers.make_square(mirseq, seq).tolist())


                # batch_tpm_kd_x_original = np.array(batch_tpm_kd_x).reshape([-1, 4*MIRLEN, 4*SEQLEN, 1])

                # print(time.time() - t0)
                # ######################

                # t0 = time.time()
                batch_tpm_kd_x = np.zeros([(BATCH_SIZE_REPRESSION * NUM_TRAIN * num_sites) + BATCH_SIZE_BIOCHEM, 4*MIRLEN, 4*SEQLEN])
                batch_tpm_mask = np.zeros([BATCH_SIZE_REPRESSION, NUM_TRAIN, num_sites])
                for counter1, big_seq_list in enumerate(all_seqs):

                    for counter2, (mir, seq_list) in enumerate(zip(train_mirs, big_seq_list)):

                        if len(seq_list) == 0:
                            continue

                        mirseq = MIRSEQ_DICT_MIRLEN[mir]
                        current = (counter1 * NUM_TRAIN * num_sites) + (counter2 * num_sites)
                        for seq in seq_list:
                            batch_tpm_kd_x[current, :, :] = helpers.make_square(mirseq, seq)
                            current += 1
                        batch_tpm_mask[counter1, counter2, :len(seq_list)] = 1.0

                batch_tpm_y = repression_train_batch[train_mirs].values

                current = BATCH_SIZE_REPRESSION * NUM_TRAIN * num_sites
                for mirseq, seq in zip(biochem_train_batch['mirseq'], biochem_train_batch['seq']):
                    batch_tpm_kd_x[current, :, :] = helpers.make_square(mirseq, seq)
                    current += 1

                batch_tpm_kd_x = np.expand_dims(batch_tpm_kd_x, 3)
                batch_kd_y = biochem_train_batch[['log ka']].values

                # print(time.time() - t0)

                # print(np.sum(np.abs(batch_tpm_kd_x - batch_tpm_kd_x_original)))
                # print(np.sum(np.abs(batch_tpm_kd_x[(-1 * BATCH_SIZE_BIOCHEM):, :, :, :] - batch_tpm_kd_x_original[(-1 * BATCH_SIZE_BIOCHEM):, :, :, :])))
                # print(np.sum(np.abs(batch_tpm_mask - batch_tpm_mask_original)))

                # make feed dict for training
                feed_dict = {
                        _keep_prob: KEEP_PROB_TRAIN,
                        _phase_train: True,
                        _kd_x: batch_tpm_kd_x,
                        _kd_y: batch_kd_y,
                        _tpm_mask: batch_tpm_mask,
                        _tpm_y: batch_tpm_y
                    }

                # print(batch_kd_y, batch_tpm_y, batch_tpm_mask)
                # stuffs = sess.run([_pred_kd_ind, _pred_kd, _pred_kd_tpm_flat,
                #                  _pred_kd_tpm, _blah, _blah2, _pred_nbound, _pred_tpm, _kd_loss, _tpm_loss], feed_dict=feed_dict)

                # for stuff in stuffs:
                #     print(stuff)
                # break

                # run train step
                _, l1, l2, l3 = sess.run([_train_step, _kd_loss, _tpm_loss, _weight_regularize], feed_dict=feed_dict)


                if (step % REPORT_INT) == 0:

                    saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=step)


                    print(l1, l2, l3)
                    step_list.append(step)
                    train_losses.append(l1+l2+l3)

                    feed_dict = {
                        _keep_prob: 1.0,
                        _phase_train: False,
                        _kd_x: batch_tpm_kd_x,
                        _kd_y: batch_kd_y,
                        _tpm_mask: batch_tpm_mask,
                        _tpm_y: batch_tpm_y
                    }

                    train_kd_preds = sess.run(_pred_kd, feed_dict=feed_dict)

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(train_kd_preds.flatten(), batch_kd_y.flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'train_kd_scatter.png'))
                    plt.close()

                    train_tpm_preds = sess.run(_pred_tpm, feed_dict=feed_dict)

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(train_tpm_preds, batch_tpm_y)
                    plt.savefig(os.path.join(options.LOGDIR, 'train_tpm_scatter.png'))
                    plt.close()


                    conv_weights = sess.run(_w1)
                    xlabels = ['U','A','G','C']
                    ylabels = ['A','U','C','G']
                    helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1.pdf'))

                    current_freeAGO = np.mean(sess.run(_freeAGO))
                    current_slope = sess.run(_slope)
                    print('current free AGO: {:.3}'.format(current_freeAGO))
                    print('current slope: {:.3}'.format(current_slope))

                    pred_nbound_test = []
                    pred_test_mir_kds = []
                    pred_test_mir_seqs = []
                    for seq_list in test_seqs:

                        pred_test_mir_seqs += seq_list

                        if len(seq_list) == 0:
                            pred_nbound_test.append(0.0)
                            continue 

                        batch_tpm_kd_x = [helpers.make_square(test_mirseq, seq) for seq in seq_list]
                        batch_tpm_kd_x = np.array(batch_tpm_kd_x).reshape([-1, 4*MIRLEN, 4*SEQLEN, 1])

                        feed_dict = {
                                        _keep_prob: 1.0,
                                        _phase_train: False,
                                        _kd_x: batch_tpm_kd_x
                                    }

                        pred_kd_test = sess.run(_pred_kd_ind, feed_dict=feed_dict)

                        pred_test_mir_kds += list(pred_kd_test.flatten())
                        pred_nbound_test.append(np.sum(1.0 / (1.0 + np.exp(-1*pred_kd_test - current_freeAGO))))

                    pred_nbound_test = np.array(pred_nbound_test)
                    print(stats.linregress(pred_nbound_test.flatten(), test_logfc_labels.flatten()))

                    test_losses.append(np.sum(((pred_nbound_test.flatten() * current_slope) - test_logfc_labels.flatten())**2)/len(test_logfc_labels))

                    fig = plt.figure(figsize=(7,5))
                    plt.plot(step_list, test_losses)
                    plt.savefig(os.path.join(options.LOGDIR, 'test_losses.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7,5))
                    plt.plot(step_list, train_losses)
                    plt.savefig(os.path.join(options.LOGDIR, 'train_losses.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(pred_nbound_test.flatten(), test_logfc_labels.flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'test_scatter.png'))
                    plt.close()


                    # actual_test_mir_kds = data[data['mir'] == options.TEST_MIRNA].set_index('seq').loc[pred_test_mir_seqs]['log ka']
                    # fig = plt.figure(figsize=(7,7))
                    # plt.scatter(pred_test_mir_kds, actual_test_mir_kds)
                    # plt.savefig(os.path.join(options.LOGDIR, 'test_scatter_kds.png'))
                    # plt.close()

                    fig = plt.figure(figsize=(7,7))
                    plt.hist(pred_test_mir_kds, bins=100)
                    plt.savefig(os.path.join(options.LOGDIR, 'test_kds_hist.png'))
                    plt.close()

                step += 1

                    








