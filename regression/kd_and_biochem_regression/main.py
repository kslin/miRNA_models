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

def get_tpm_seqs(utr, mirs):
    all_seqs = []
    num_sites = 0
    for mir in mirs:
        site = helpers.SITE_DICT[mir]
        locs1 = [m.start() for m in re.finditer(site[:-1], utr)]
        locs2 = [(m.start() - 1) for m in re.finditer(site[1:], utr)]
        locs = list(set(locs1 + locs2))
        seqs = [utr[loc-4:loc+8] for loc in locs if (loc-4 >=0) and (loc+8 <= len(utr))]
        if len(seqs) > num_sites:
            num_sites = len(seqs)
        all_seqs.append(seqs)

    return num_sites, all_seqs


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
    BATCH_SIZE = 100
    KEEP_PROB_TRAIN = 0.5
    STARTING_LEARNING_RATE = 0.001
    LAMBDA = 0.005
    NUM_EPOCHS = 2
    REPORT_INT = 250

    HIDDEN1 = 4
    HIDDEN2 = 8
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

    train_mirs = [m for m in MIRS if m != options.TEST_MIRNA]
    test_mirs = [options.TEST_MIRNA]
    print(train_mirs)
    print(test_mirs)
    NUM_TRAIN = len(train_mirs)
    NUM_TEST = len(test_mirs)

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
    

    ### DEFINE MODEL ###

    # reset and build the neural network
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        # phase_train2 = tf.placeholder(tf.bool, name='phase_train2')
        # gene_num = tf.placeholder(tf.float32, shape=[NUM_GENES], name='gene_num')

        kd_x = tf.placeholder(tf.float32, shape=[None, 4 * MIRLEN, 4 * SEQLEN, 1], name='kd_x')
        kd_y = tf.placeholder(tf.float32, shape=[None, 1], name='kd_y')
        tpm_x = tf.placeholder(tf.float32, shape=[None, 4 * MIRLEN, 4 * SEQLEN, 1], name='tpm_x')
        tpm_mask = tf.placeholder(tf.float32, shape=[None, None], name='tpm_mask')
        tpm_y = tf.placeholder(tf.float32, shape=[None], name='tpm_y')

        freeAGO = tf.get_variable('freeAGO', shape=[NUM_TRAIN,1], initializer=tf.constant_initializer(-5.0), trainable=False)
        slope = tf.get_variable('slope', shape=(), initializer=tf.constant_initializer(-0.51023716), trainable=False)
        # intercept = tf.get_variable('intercept', shape=[NUM_GENES],
        #                             initializer=tf.constant_initializer(baseline_init), trainable=False)

        with tf.name_scope('layer1'):
            w1, b1 = get_conv_params(4, 4, 1, HIDDEN1, 'layer1')
            preactivate1_kd = tf.nn.conv2d(kd_x, w1, strides=[1, 4, 4, 1], padding='VALID') + b1
            preactivate1_tpm = tf.nn.conv2d(tpm_x, w1, strides=[1, 4, 4, 1], padding='VALID') + b1

            # preactivate1_kd_bn = tf.layers.batch_normalization(preactivate1_kd, axis=1, training=phase_train)
            # preactivate1_tpm_bn = tf.layers.batch_normalization(preactivate1_tpm, axis=1, training=phase_train)

            layer1_kd = tf.nn.relu(preactivate1_kd)
            layer1_tpm = tf.nn.relu(preactivate1_tpm)

        with tf.name_scope('layer2'):
            w2, b2 = get_conv_params(2, 2, HIDDEN1, HIDDEN2, 'layer2')
            preactivate2_kd = tf.nn.conv2d(layer1_kd, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
            preactivate2_tpm = tf.nn.conv2d(layer1_tpm, w2, strides=[1, 1, 1, 1], padding='SAME') + b2

            # preactivate2_kd_bn = tf.layers.batch_normalization(preactivate2_kd, axis=1, training=phase_train)
            # preactivate2_tpm_bn = tf.layers.batch_normalization(preactivate2_tpm, axis=1, training=phase_train)

            layer2_kd = tf.nn.relu(preactivate2_kd)
            layer2_tpm = tf.nn.relu(preactivate2_tpm)

        with tf.name_scope('layer3'):
            w3, b3 = get_conv_params(MIRLEN, SEQLEN, HIDDEN2, HIDDEN3, 'layer3')
            preactivate3_kd = tf.nn.conv2d(layer2_kd, w3, strides=[1, MIRLEN, SEQLEN, 1], padding='VALID') + b3
            preactivate3_tpm = tf.nn.conv2d(layer2_tpm, w3, strides=[1, MIRLEN, SEQLEN, 1], padding='VALID') + b3

            # preactivate3_kd_bn = tf.layers.batch_normalization(preactivate3_kd, axis=1, training=phase_train)
            # preactivate3_tpm_bn = tf.layers.batch_normalization(preactivate3_tpm, axis=1, training=phase_train)

            layer3_kd = tf.nn.relu(preactivate3_kd)
            layer3_tpm = tf.nn.relu(preactivate3_tpm)

        # add dropout
        with tf.name_scope('dropout'):
            dropout_kd = tf.nn.dropout(layer3_kd, keep_prob)
            dropout_tpm = tf.nn.dropout(layer3_tpm, keep_prob)

        # reshape to 1D tensor
        layer_flat_kd = tf.reshape(dropout_kd, [-1, HIDDEN3])
        layer_flat_tpm = tf.reshape(dropout_tpm, [-1, HIDDEN3])

        # add last layer
        with tf.name_scope('final_layer'):
            with tf.name_scope('weights'):
                w4 = weight_variable([HIDDEN3, 1], HIDDEN3)

                # add variable to collection of variables
                tf.add_to_collection('weight', w4)
            with tf.name_scope('biases'):
                b4 = bias_variable([1])

                # add variable to collection of variables
                tf.add_to_collection('bias', b4)

            pred_kd = tf.matmul(layer_flat_kd, w4) + b4
            pred_kd_ind = tf.matmul(layer_flat_tpm, w4) + b4
            pred_kd_ind_flat = tf.reshape(pred_kd_ind, [NUM_TRAIN, -1])

        pred_nbound = tf.reduce_sum(tf.multiply(tf.nn.sigmoid(freeAGO - pred_kd_ind_flat), tpm_mask), axis=1)
        pred_tpm = (pred_nbound * slope)
            

        weight_regularize = tf.multiply(tf.nn.l2_loss(w1) \
                                + tf.nn.l2_loss(w2) \
                                + tf.nn.l2_loss(w3) \
                                + tf.nn.l2_loss(w4), LAMBDA)


        kd_loss = tf.nn.l2_loss(tf.subtract(pred_kd, kd_y)) / BATCH_SIZE
        tpm_loss = 100 * tf.nn.l2_loss(tf.subtract(pred_tpm, tpm_y)) / NUM_TRAIN

        loss = kd_loss + tpm_loss + weight_regularize

        train_step = tf.train.AdamOptimizer(STARTING_LEARNING_RATE).minimize(loss)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_step = tf.train.AdamOptimizer(STARTING_LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # TOY DATA
        # toy_mirseq = 'GGATGATGGAGA'
        # toy_seqs = ['CCTACTACCTCT',
        #             'GAATCTACCTCT',
        #             'GTACGATCGATC']

        # toy_kd_y = [[-5.0], [-4.0], [-1.0]]
        # toy_tpm_y = [-1.0, -2.0]
        # toy_kd_x = np.array([helpers.make_square(toy_mirseq, t) for t in toy_seqs]).reshape(3,48,48,1)
        # toy_tpm_x = np.array([helpers.make_square(toy_mirseq, toy_seqs[0]), helpers.make_square(toy_mirseq, toy_seqs[1]),
        #                       helpers.make_square(toy_mirseq, toy_seqs[2]), np.zeros((48,48)).tolist()]).reshape(4,48,48,1)
        # toy_tpm_mask = np.array([[1,1],[1,0]])
        # feed_dict = {
        #                 keep_prob: KEEP_PROB_TRAIN,
        #                 phase_train: True,
        #                 gene_num: np.array([0,1,0]),
        #                 kd_x: toy_kd_x,
        #                 kd_y: toy_kd_y,
        #                 tpm_x: toy_tpm_x,
        #                 tpm_mask: toy_tpm_mask,
        #                 tpm_y: toy_tpm_y
        #             }
        # a1, a2, f = sess.run([pred_kd_ind, pred_nbound, pred_tpm], feed_dict=feed_dict)
        # a3 = 1.0 / (1.0 + np.exp(a1 + 5))
        # print(a1)
        # print(a3)
        # print(a2)
        # print((a2 * -1) + baseline_init[1])
        # print(f)

        kd_ix = 0
        for epoch_counter in range(NUM_EPOCHS):

            for gene_ix, tpm_row in enumerate(train_tpm.iterrows()):

                num_sites, all_seqs = get_tpm_seqs(tpm_row[1]['Sequence'], train_mirs)
                if num_sites == 0:
                    continue

                batch_tpm_x = []
                batch_tpm_mask = []
                for mir, seq_list in zip(train_mirs, all_seqs):
                    mask_temp = []
                    mirseq = helpers.MIRSEQ_DICT[mir][:MIRLEN][::-1]
                    for seq in seq_list:
                        batch_tpm_x.append(helpers.make_square(mirseq, seq))
                        mask_temp.append(1.0)
                    for _ in range(num_sites - len(seq_list)):
                        batch_tpm_x.append(np.zeros((4*MIRLEN,4*SEQLEN)).tolist())
                        mask_temp.append(0.0)
                    batch_tpm_mask.append(mask_temp)

                batch_tpm_x = np.array(batch_tpm_x).reshape(NUM_TRAIN*num_sites, 4*MIRLEN,4*SEQLEN, 1)
                batch_tpm_mask = np.array(batch_tpm_mask)
                batch_tpm_y = tpm_row[1][train_mirs].values.reshape([NUM_TRAIN])
                batch_gene_num = np.zeros(NUM_GENES)
                batch_gene_num[gene_ix] = 1.0

                # print(batch_tpm_x.shape, batch_tpm_mask.shape, batch_tpm_y.shape)
                # print(np.sum(batch_gene_num))


                if (len(data_temp) - kd_ix) < BATCH_SIZE:
                    shuffle_ix = np.random.permutation(len(data))
                    data_temp = data.iloc[shuffle_ix]
                    data_temp['keep'] = [(np.random.random() > 0.9) if x == 'grey' else True for x in data_temp['color']]
                    data_temp = data_temp[data_temp['keep']]
                    print('new epoch')
                    kd_ix = 0

                kd_subdf = data_temp.iloc[kd_ix: kd_ix+BATCH_SIZE]
                kd_ix += BATCH_SIZE

                batch_kd_x = []
                batch_kd_y = []
                for row in kd_subdf.iterrows():
                    batch_kd_x.append(helpers.make_square(row[1]['mirseq'], row[1]['seq']))
                    batch_kd_y.append(row[1]['log kd'])

                batch_kd_x = np.array(batch_kd_x).reshape(BATCH_SIZE, 4*MIRLEN, 4*SEQLEN, 1)
                batch_kd_y = np.array(batch_kd_y).reshape(BATCH_SIZE, 1)

                # print(batch_kd_x.shape, batch_kd_y.shape)

                feed_dict = {
                        keep_prob: KEEP_PROB_TRAIN,
                        phase_train: True,
                        # gene_num: batch_gene_num,
                        kd_x: batch_kd_x,
                        kd_y: batch_kd_y,
                        tpm_x: batch_tpm_x,
                        tpm_mask: batch_tpm_mask,
                        tpm_y: batch_tpm_y
                    }

                _, l1, l2, l3 = sess.run([train_step, kd_loss, tpm_loss, weight_regularize], feed_dict=feed_dict)

                if (gene_ix % REPORT_INT) == 0:
                    print(l1, l2, l3)

                    train_kd_preds = sess.run(pred_kd, feed_dict=feed_dict)

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(train_kd_preds.flatten(), batch_kd_y.flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'train_scatter.png'))
                    plt.close()


                    conv_weights = sess.run(w1)
                    xlabels = ['U','A','G','C']
                    ylabels = ['A','U','C','G']
                    helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1.pdf'))

                    pred_nbound_test = []
                    current_freeAGO = np.mean(sess.run(freeAGO))
                    current_slope = sess.run(slope)
                    print('current free AGO: {:.3}'.format(current_freeAGO))
                    print('current slope: {:.3}'.format(current_slope))


                    pred_test_mir_kds = []
                    pred_test_mir_seqs = []
                    for seq_list in test_seqs:

                        pred_test_mir_seqs += seq_list

                        if len(seq_list) == 0:
                            pred_nbound_test.append(0.0)
                            continue 

                        batch_tpm_x = []
                        for seq in seq_list:
                            batch_tpm_x.append(helpers.make_square(test_mirseq, seq))

                        batch_tpm_x = np.array(batch_tpm_x).reshape(len(seq_list), 4*MIRLEN,4*SEQLEN, 1)

                        feed_dict = {
                                        keep_prob: 1.0,
                                        phase_train: False,
                                        tpm_x: batch_tpm_x
                                    }

                        pred_kd_test = sess.run(pred_kd_ind, feed_dict=feed_dict)
                        pred_test_mir_kds += list(pred_kd_test.flatten())
                        pred_nbound_test.append(np.sum(1.0 / (1.0 + np.exp(pred_kd_test - current_freeAGO))))

                    pred_nbound_test = np.array(pred_nbound_test)
                    print(stats.linregress(pred_nbound_test.flatten(), test_logfc_labels.flatten()))

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(pred_nbound_test.flatten(), test_logfc_labels.flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'test_scatter.png'))
                    plt.close()


                    actual_test_mir_kds = data[data['mir'] == options.TEST_MIRNA].set_index('seq').loc[pred_test_mir_seqs]['log kd']
                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(pred_test_mir_kds, actual_test_mir_kds)
                    plt.savefig(os.path.join(options.LOGDIR, 'test_scatter_kds.png'))
                    plt.close()

                    # fig = plt.figure(figsize=(7,7))
                    # plt.hist(pred_test_mir_kds, bins=100)
                    # plt.savefig(os.path.join(options.LOGDIR, 'test_kds_hist.png'))
                    # plt.close()

                    








