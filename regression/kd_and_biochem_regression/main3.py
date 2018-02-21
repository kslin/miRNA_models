from optparse import OptionParser
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
# import seaborn as sns
import tensorflow as tf

import helpers

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


def weight_variable(shape, n_in, name=None):
    # print(n_in)
    # initial = tf.random_normal(shape, stddev=np.sqrt(2/n_in))
    initial = tf.truncated_normal(shape, stddev=0.1)
    # initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def get_conv_params(dim1, dim2, in_channels, out_channels, layer_name):

    # create variables for weights and biases
    with tf.name_scope('weights'):
        weights = weight_variable([dim1, dim2, in_channels, out_channels], dim1*dim2, name="{}_weight".format(layer_name))

        # add variable to collection of variables
        tf.add_to_collection('weight', weights)
    with tf.name_scope('biases'):
        biases = bias_variable([out_channels], name="{}_bias".format(layer_name))

        # add variable to collection of variables
        tf.add_to_collection('bias', biases)

    return weights, biases


def get_seqs(utr, site):
    locs1 = [m.start() for m in re.finditer(site[:-1], utr)]
    locs2 = [(m.start() - 1) for m in re.finditer(site[1:], utr)]
    locs = list(set(locs1 + locs2))
    seqs = [utr[loc-4:loc+8] for loc in locs if (loc-4 >=0) and (loc+8 <= len(utr))]
    return seqs

def get_tpm_seqs(utr, mirs):
    all_seqs = []
    num_sites = 0
    for mir in mirs:
        site = helpers.SITE_DICT[mir]
        seqs = get_seqs(utr, site)
        if len(seqs) > num_sites:
            num_sites = len(seqs)
        all_seqs.append(seqs)

    return num_sites, all_seqs


def boolean_indexing(v, mirlen, seqlen):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape, np.zeros((4*mirlen, 4*seqlen, 1)).tolist(), dtype=object)
    out[mask] = np.concatenate(v)
    out = np.array(out.tolist(), dtype=float)
    return out, mask


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-k", "--kdfile", dest="KD_FILE", help="kd data")
    parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    parser.add_option("-i", "--init_baselines", dest="BASELINE_FILE", help="initial baseline data")
    parser.add_option("-m", "--mirna", dest="TEST_MIRNA", help="testing miRNA")
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs", default=None)

    (options, args) = parser.parse_args()

    MIRLEN = 12
    SEQLEN = 12
    BATCH_SIZE_KD = 200
    BATCH_SIZE_TPM = 10
    KEEP_PROB_TRAIN = 0.5
    STARTING_LEARNING_RATE = 0.002
    LAMBDA = 0.00
    NUM_EPOCHS = 10
    REPORT_INT = 50
    TPM_WEIGHT = 5

    HIDDEN1 = 2
    HIDDEN2 = 4
    HIDDEN3 = 16

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

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
    data['log ka'] = -1 * data['log kd']
    data['mirseq'] = [x[:MIRLEN][::-1] for x in data['mirseq_full']]
    data['sitem8'] = [helpers.rev_comp(mirseq[1:8]) for mirseq in data['mirseq_full']]
    data['color'] = [helpers.get_color_old(sitem8, seq) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]
    data['color2'] = [helpers.get_color_old(sitem8, seq[2:10]) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

    # get rid of sequences with sites out of register
    print(len(data))
    data = data[data['color'] == data['color2']].drop('color2',1)
    print(len(data))

    # shuffle data
    shuffle_ix = np.random.permutation(len(data))
    data_temp = data.iloc[shuffle_ix]
    data_temp['keep'] = [(np.random.random() > 0.9) if x == 'grey' else True for x in data_temp['color']]
    data_temp = data_temp[data_temp['keep']]

    # ## READ INITIAL BASELINE ###
    baseline_init = pd.read_csv(options.BASELINE_FILE, sep='\t', index_col=0)
    assert (len(baseline_init) == len(tpm))
    NUM_GENES = len(baseline_init)
    baseline_init = baseline_init.loc[tpm.index]['nosite_tpm'].values.reshape([NUM_GENES, 1])
    # baseline_init = baseline_init['nosite_tpm'].values.reshape([NUM_GENES])

    train_tpm[train_mirs] = train_tpm[train_mirs].values - baseline_init
    test_tpm[test_mirs] = test_tpm[test_mirs].values - baseline_init

    subset = np.random.choice(np.arange(len(test_tpm)), size=400)
    test_tpm = test_tpm.iloc[subset]
    test_logfc_labels = test_tpm[test_mirs].values

    test_mirseq = helpers.MIRSEQ_DICT[options.TEST_MIRNA][:MIRLEN][::-1]
    test_seqs = []
    for utr in test_tpm['Sequence']:
        _, seqs = get_tpm_seqs(utr, [options.TEST_MIRNA])
        test_seqs.append(seqs[0])


    shuffle_ix = np.random.permutation(len(train_tpm))
    train_tpm_temp = train_tpm.iloc[shuffle_ix]
    

    ### DEFINE MODEL ###

    # reset and build the neural network
    tf.reset_default_graph()

    _keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    _phase_train = tf.placeholder(tf.bool, name='phase_train')

    _kd_x = tf.placeholder(tf.float32, shape=[None, 4 * MIRLEN, 4 * SEQLEN, 1], name='kd_x')
    _kd_y = tf.placeholder(tf.float32, shape=[None, 1], name='kd_y')
    _kd_mask = tf.placeholder(tf.float32, shape=[None, 1], name='kd_mask')
    _tpm_mask = tf.placeholder(tf.float32, shape=[None, None, None], name='tpm_mask')
    _tpm_y = tf.placeholder(tf.float32, shape=[None, None], name='tpm_y') 

    _freeAGO = tf.get_variable('freeAGO', shape=[1,NUM_TRAIN,1], initializer=tf.constant_initializer(-5.0), trainable=False)
    _slope = tf.get_variable('slope', shape=(), initializer=tf.constant_initializer(-0.51023716), trainable=False)
    # intercept = tf.get_variable('intercept', shape=[NUM_GENES],
    #                             initializer=tf.constant_initializer(baseline_init), trainable=False)

    with tf.name_scope('layer1'):
        _w1, _b1 = get_conv_params(4, 4, 1, HIDDEN1, 'layer1')
        _preactivate1_kd = tf.nn.conv2d(_kd_x, _w1, strides=[1, 4, 4, 1], padding='VALID') + _b1

        _preactivate1_kd_bn = tf.layers.batch_normalization(_preactivate1_kd, axis=1, training=_phase_train)

        _layer1_kd = tf.nn.relu(_preactivate1_kd_bn)

    with tf.name_scope('layer2'):
        _w2, _b2 = get_conv_params(2, 2, HIDDEN1, HIDDEN2, 'layer2')
        _preactivate2_kd = tf.nn.conv2d(_layer1_kd, _w2, strides=[1, 1, 1, 1], padding='SAME') + _b2

        _preactivate2_kd_bn = tf.layers.batch_normalization(_preactivate2_kd, axis=1, training=_phase_train)

        _layer2_kd = tf.nn.relu(_preactivate2_kd_bn)

    with tf.name_scope('layer3'):
        _w3, _b3 = get_conv_params(MIRLEN, SEQLEN, HIDDEN2, HIDDEN3, 'layer3')
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
            _w4 = weight_variable([HIDDEN3, 1], HIDDEN3)

            # add variable to collection of variables
            tf.add_to_collection('weight', _w4)
        with tf.name_scope('biases'):
            _b4 = bias_variable([1])

            # add variable to collection of variables
            tf.add_to_collection('bias', _b4)

        # split into kd outputs and tpm outputs
        _pred_kd_ind = tf.matmul(_layer_flat_kd, _w4) + _b4
        _pred_kd = _pred_kd_ind[-1 * BATCH_SIZE_KD:, :1]
        _pred_kd_tpm_flat = _pred_kd_ind[:-1 * BATCH_SIZE_KD, :1]
        _pred_kd_tpm = tf.reshape(_pred_kd_tpm_flat, [BATCH_SIZE_TPM, NUM_TRAIN, -1])

    _pred_nbound = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(_freeAGO + _pred_kd_tpm), _tpm_mask), axis=2)
    _pred_tpm = (_pred_nbound * _slope)
        

    _weight_regularize = tf.multiply(tf.nn.l2_loss(_w1) \
                            + tf.nn.l2_loss(_w2) \
                            + tf.nn.l2_loss(_w3) \
                            + tf.nn.l2_loss(_w4), LAMBDA)

    _kd_loss = tf.nn.l2_loss(tf.subtract(_pred_kd, _kd_y)) / BATCH_SIZE_KD
    _tpm_loss = TPM_WEIGHT * tf.nn.l2_loss(tf.subtract(_pred_tpm, _tpm_y)) / NUM_TRAIN

    _loss = _kd_loss + _tpm_loss + _weight_regularize
    # _loss = _kd_loss + _weight_regularize

    # train_step = tf.train.AdamOptimizer(STARTING_LEARNING_RATE).minimize(loss)

    _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(_update_ops):
        _train_step = tf.train.AdamOptimizer(STARTING_LEARNING_RATE).minimize(_loss)

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:
        sess.run(tf.global_variables_initializer())

        kd_ix = 0
        tpm_ix = 0
        epoch_counter = 1

        step = 0
        while True:

            if (len(train_tpm) - tpm_ix) < BATCH_SIZE_TPM:
                shuffle_ix = np.random.permutation(len(train_tpm))
                train_tpm_temp = train_tpm.iloc[shuffle_ix]
                epoch_counter += 1
                if epoch_counter > NUM_EPOCHS:
                    break
                tpm_ix = 0

            train_tpm_subdf = train_tpm_temp.iloc[tpm_ix: tpm_ix + BATCH_SIZE_TPM]
            tpm_ix += BATCH_SIZE_TPM

            all_seqs = []
            num_sites = 0
            # all_num_sites = []
            for tpm_row in train_tpm_subdf.iterrows():

                num_sites_gene, gene_seqs = get_tpm_seqs(tpm_row[1]['Sequence'], train_mirs)
                all_seqs.append(gene_seqs)
                # all_num_sites.append(num_sites_gene)
                if num_sites_gene > num_sites:
                    num_sites = num_sites_gene

            batch_tpm_kd_x = []
            batch_tpm_mask = []
            batch_tpm_y = []
            for big_seq_list in all_seqs:

                big_mask_temp = []
                for mir, seq_list in zip(train_mirs, big_seq_list):
                    mirseq = helpers.MIRSEQ_DICT[mir][:MIRLEN][::-1]
                    mask_temp = []
                    for seq in seq_list:
                        batch_tpm_kd_x.append(helpers.make_square(mirseq, seq))
                        mask_temp.append(1.0)
                    for _ in range(num_sites - len(seq_list)):
                        batch_tpm_kd_x.append(np.zeros((4*MIRLEN,4*SEQLEN,1)).tolist())
                        mask_temp.append(0.0)

                    big_mask_temp.append(mask_temp)

                batch_tpm_mask.append(big_mask_temp)

            batch_tpm_mask = np.array(batch_tpm_mask)
            batch_tpm_y = train_tpm_subdf[train_mirs].values


            if (len(data_temp) - kd_ix) < BATCH_SIZE_KD:
                shuffle_ix = np.random.permutation(len(data))
                data_temp = data.iloc[shuffle_ix]
                data_temp['keep'] = [(np.random.random() > 0.9) if x == 'grey' else True for x in data_temp['color']]
                data_temp = data_temp[data_temp['keep']]
                print('new epoch')
                kd_ix = 0

            kd_subdf = data_temp.iloc[kd_ix: kd_ix+BATCH_SIZE_KD]
            kd_ix += BATCH_SIZE_KD

            batch_kd_y = []
            for row in kd_subdf.iterrows():
                batch_tpm_kd_x.append(helpers.make_square(row[1]['mirseq'], row[1]['seq']))
                batch_kd_y.append(row[1]['log ka'])


            batch_tpm_kd_x = np.array(batch_tpm_kd_x)
            batch_kd_y = np.array(batch_kd_y).reshape(BATCH_SIZE_KD, 1)

            # print(batch_tpm_kd_x.shape, batch_kd_y.shape, batch_tpm_mask.shape, batch_tpm_y.shape)


            feed_dict = {
                    _keep_prob: KEEP_PROB_TRAIN,
                    _phase_train: True,
                    _kd_x: batch_tpm_kd_x,
                    _kd_y: batch_kd_y,
                    _tpm_mask: batch_tpm_mask,
                    _tpm_y: batch_tpm_y
                }

            _, l1, l2, l3 = sess.run([_train_step, _kd_loss, _tpm_loss, _weight_regularize], feed_dict=feed_dict)


            if (step % REPORT_INT) == 0:
                print(l1, l2, l3)

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
                for blah, seq_list in enumerate(test_seqs):

                    pred_test_mir_seqs += seq_list

                    if len(seq_list) == 0:
                        pred_nbound_test.append(0.0)
                        continue 

                    batch_tpm_kd_x = []
                    for seq in seq_list:
                        batch_tpm_kd_x.append(helpers.make_square(test_mirseq, seq))

                    batch_tpm_kd_x = np.array(batch_tpm_kd_x)

                    feed_dict = {
                                    _keep_prob: 1.0,
                                    _phase_train: False,
                                    _kd_x: batch_tpm_kd_x
                                }

                    pred_kd_test = sess.run(_pred_kd_ind, feed_dict=feed_dict)

                    # if blah == 1:
                    #     print(seq_list)
                    #     print(batch_tpm_kd_x.shape)
                    #     print(pred_kd_test)
                    #     print(data[data['mir'] == options.TEST_MIRNA].set_index('seq').loc[seq_list]['log ka'])

                    pred_test_mir_kds += list(pred_kd_test.flatten())
                    pred_nbound_test.append(np.sum(1.0 / (1.0 + np.exp(-1*pred_kd_test - current_freeAGO))))

                pred_nbound_test = np.array(pred_nbound_test)
                print(stats.linregress(pred_nbound_test.flatten(), test_logfc_labels.flatten()))

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

                    








