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

import config, helpers, data_objects_endog, tf_helpers

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-k", "--kdfile", dest="KD_FILE", help="kd data")
    parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    parser.add_option("-m", "--mirna", dest="TEST_MIRNA", help="testing miRNA")
    parser.add_option("--pretrain", dest="PRETRAIN", help="pretrain directory")
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs")

    (options, args) = parser.parse_args()

    PRETRAIN_SAVE_PATH = os.path.join(options.LOGDIR, 'pretrain_saved')
    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    # make sure config.BATCH_SIZE_BIOCHEM is an even number
    # if config.BATCH_SIZE_BIOCHEM % 4 != 0:
    #     raise ValueError("config.BATCH_SIZE_BIOCHEM must be a multiple of 4")

    ### READ EXPRESSION DATA ###
    tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)
    MIRS = [x for x in tpm.columns if ('mir' in x) or ('lsy' in x)]

    # split miRNAs into training and testing
    if options.TEST_MIRNA == 'none':
        train_mirs = MIRS
        test_mirs = ['mir139']
        TEST_MIRNA = 'mir139'
    else:
        assert options.TEST_MIRNA in (MIRS + ['let7'])
        TEST_MIRNA = options.TEST_MIRNA
        train_mirs = [m for m in MIRS if m != TEST_MIRNA]
        test_mirs = [TEST_MIRNA]

    print('Train miRNAs: {}'.format(train_mirs))
    print('Test miRNAs: {}'.format(test_mirs))
    NUM_TRAIN = len(train_mirs)
    NUM_TEST = len(test_mirs)

    print(NUM_TRAIN, NUM_TEST)

    tpm = tpm.rename(columns={'sequence': 'Sequence'})

    if config.BASELINE_METHOD is not None:
        # split tpm data into training and testing
        train_tpm = tpm[train_mirs + [config.BASELINE_METHOD, 'Sequence']].dropna(subset=[config.BASELINE_METHOD])
        test_tpm = tpm[test_mirs + [config.BASELINE_METHOD, 'Sequence']]

    else:
        # split tpm data into training and testing
        train_tpm = tpm[train_mirs + ['Sequence']]
        test_tpm = tpm[test_mirs + ['nosite3', 'Sequence']]

    sitem8 = config.SITE_DICT[TEST_MIRNA][:-1]
    test_tpm['num_canon'] = [helpers.count_num_canon(utr, sitem8) for utr in test_tpm['Sequence']]

    ### READ KD DATA ###
    data = pd.read_csv(options.KD_FILE, sep='\t')
    data.columns = ['seq','log_kd','mir','mirseq_full','stype']
    # data.columns = ['mir','mirseq_full','seq','log_kd','stype']

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

    # zero-center and normalize Ka's
    if config.ONLY_CANON:
        data = data[data['stype'] != 'no site']
        data['keep_prob'] = 1.0
    else:
        # data['keep_prob'] = (1 / (1 + np.exp((data['log_kd'] + 1)*4)))

        # def get_keep_prob(val, all_bins_dict):
        #     if val <= -4:
        #         return 1.0

        #     else:
        #         return all_bins_dict[val]
        
        # data['nearest'] = np.minimum(0, np.round(data['log_kd']*4)/4)
        # all_bins = data.copy()
        # all_bins['count'] = 1
        # all_bins = all_bins.groupby('nearest').agg({'count': np.sum})

        # temp_for_linregress = all_bins[all_bins.index <= -4]
        # slope, intercept = stats.linregress(temp_for_linregress.index, temp_for_linregress['count'])[:2]
        # all_bins['final_count'] = all_bins.index * slope + intercept

        # some_bins = all_bins[all_bins.index >= -4]
        # some_bins['keep_prob'] = some_bins['final_count'] / some_bins['count']
        # some_bins_dict = {x:y for (x,y) in zip(some_bins.index, some_bins['keep_prob'])}
        # data['keep_prob'] = [get_keep_prob(x, some_bins_dict) for x in data['nearest']]

        print("Balancing data...")
        data['nearest'] = np.minimum(0, np.round(data['log_kd']*4)/4)
        data['count'] = 1
        temp = data.groupby('nearest').agg({'count': np.sum})
        temp['target'] = np.exp(temp.index + 5)*500
        temp['keep_prob'] = np.minimum(1.0, temp['target'] / temp['count'])
        temp['keep_prob'] = [1.0 if x < -3 else y for (x,y) in zip(temp.index, temp['keep_prob'])]
        temp_dict = {x:y for (x,y) in zip(temp.index, temp['keep_prob'])}
        data['keep_prob'] = [temp_dict[x] for x in data['nearest']]

        data = data.drop(['nearest','count'], 1)

    temp = []
    print(len(data))
    for mir, group in data.groupby('mir'):
        site = config.SITE_DICT[mir]
        group['has_offset'] = [(((site[2:] in seq) or (site[1:-1] in seq) or (site[:-2] in seq)) and (stype == 'no site')) for (seq, stype) in zip(group['seq'], group['stype'])]
        group = group[group['has_offset'] == False]
        temp.append(group)

    data = pd.concat(temp)
    print(len(data))

    data['log ka'] = (-1.0 * data['log_kd'])
    data['mirseq'] = [config.MIRSEQ_DICT_MIRLEN[mir] for mir in data['mir']]
    data['sitem8'] = [helpers.rev_comp(mirseq[1:8]) for mirseq in data['mirseq_full']]
    data['color'] = [color_dict[stype] for stype in data['stype']]
    # data['color2'] = [helpers.get_color(sitem8, seq[2:10]) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

    # get rid of sequences with sites out of register
    print('Length of KD data: {}'.format(len(data)))
    # data = data[data['stype'] != 'no site']
    # print('Length of KD data canon sites: {}'.format(len(data)))
    # data = data[data['color'] == data['color2']].drop('color2',1)
    # print('Length of KD data, in register: {}'.format(len(data)))

    if TEST_MIRNA in data['mir'].values:
        print('Testing on {}'.format(TEST_MIRNA))
        data_test = data[data['mir'] == TEST_MIRNA]
        data_test['keep_prob'] /= 10
    else:
        print('Testing on all')
        data_test = data.copy()
        data_test['keep_prob'] /= 60

    print(len(data_test))
    data_test = data_test[[np.random.random() < x for x in data_test['keep_prob']]]
    print("Test KD miRNAs:")
    print(data_test['mir'].unique())
    print(len(data_test))

    test_kds_combined_x = np.zeros([len(data_test), 4*config.MIRLEN, 4*config.SEQLEN])
    for i, row in enumerate(data_test.iterrows()):
        mirseq_one_hot = config.ONE_HOT_DICT[row[1]['mir']]
        seq_one_hot = helpers.one_hot_encode(row[1]['seq'], config.SEQ_NT_DICT, config.TARGETS)
        test_kds_combined_x[i,:,:] = np.outer(mirseq_one_hot, seq_one_hot)
    
    test_kds_combined_x = np.expand_dims((test_kds_combined_x*4) - 0.25, 3)   
    test_kds_labels = data_test['log ka'].values
    test_kds_colors = data_test['color'].values
    
    data = data[~data['mir'].isin(test_mirs)]
    print(len(data))
    # data = data[data['log ka'] > 0]
    # data['log ka'] = np.maximum(0, data['log ka'])
    print(len(data))

    test_mir2 = np.random.choice(data['mir'].unique())
    print(test_mir2)
    data_test2 = data[data['mir'] == test_mir2]
    data_test2['keep_prob'] /= 10
    data_test2 = data_test2[[np.random.random() < x for x in data_test2['keep_prob']]]
    print(len(data_test2))

    test_kds_combined_x2 = np.zeros([len(data_test2), 4*config.MIRLEN, 4*config.SEQLEN])
    for i, row in enumerate(data_test2.iterrows()):
        mirseq_one_hot = config.ONE_HOT_DICT[row[1]['mir']]
        seq_one_hot = helpers.one_hot_encode(row[1]['seq'], config.SEQ_NT_DICT, config.TARGETS)
        test_kds_combined_x2[i,:,:] = np.outer(mirseq_one_hot, seq_one_hot)
    
    test_kds_combined_x2 = np.expand_dims((test_kds_combined_x2*4) - 0.25, 3)   
    test_kds_labels2 = data_test2['log ka'].values
    test_kds_colors2 = data_test2['color'].values


    # create data object
    biochem_train_data = data_objects_endog.BiochemData(data)
    biochem_train_data.shuffle()

    # make data objects for repression training data
    repression_train_data = data_objects_endog.RepressionData(train_tpm)
    repression_train_data.shuffle()
    repression_train_data.get_seqs(train_mirs, config.OVERLAP_DIST, config.ONLY_CANON)

    repression_test_data = data_objects_endog.RepressionData(test_tpm)
    repression_test_data.get_seqs(test_mirs, config.OVERLAP_DIST, config.ONLY_CANON)

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

        # make variable placeholders for training
        _repression_weight = tf.placeholder(tf.float32, name='repression_weight')
        _biochem_y = tf.placeholder(tf.float32, shape=[None, 1], name='biochem_y')
        _utr_len = tf.placeholder(tf.float32, shape=[None, 1], name='utr_len')
        _repression_y = tf.placeholder(tf.float32, shape=[None, None], name='repression_y')

        # add layers for predicting KA
        _pred_ind_values, _pretrain_loss, _pretrain_step, saver_pretrain, _weights, _biases = tf_helpers.make_ka_predictor(_combined_x, _pretrain_y, _keep_prob, _phase_train)
        _w1, _w2, _w3, _w4 = _weights
        _b1, _b2, _b3, _b4 = _biases
        # _w1, _w2, _w2_1, _w3, _w4 = _weights
        # _b1, _b2, _b2_1, _b3, _b4 = _biases

        init_params = [
                    config.FREEAGO_INIT,
                    config.GUIDE_OFFSET_INIT,
                    config.PASS_OFFSET_INIT,
                    config.DECAY_INIT,
                    config.UTR_COEF_INIT
                ]

        with tf.name_scope('train'):
            # split data into biochem and repression
            _repression_max_size = tf.placeholder(tf.int32, shape=[], name='repression_max_size')
            _repression_split_sizes = tf.placeholder(tf.int32, shape=[config.BATCH_SIZE_REPRESSION*NUM_TRAIN*2], name='repression_split_sizes')
            results = tf_helpers.both_steps_simple('train', NUM_TRAIN, config.BATCH_SIZE_BIOCHEM, config.BATCH_SIZE_REPRESSION, _pred_ind_values,
                                _repression_max_size, _repression_split_sizes, _utr_len, init_params)
            _pred_biochem, _pred_repression_flat, _freeAGO_mean, _freeAGO_guide_offset, _freeAGO_pass_offset, _freeAGO_all, _decay, _utr_coef, _pred_logfc = results

            # get biochem loss
            if config.BATCH_SIZE_BIOCHEM == 0:
                _biochem_loss = tf.constant(0.0)
            else:
                # _biochem_loss = (tf.nn.l2_loss(tf.subtract(_pred_biochem, _biochem_y))) / config.BATCH_SIZE_BIOCHEM
                _biochem_loss = (tf.nn.l2_loss(tf.subtract(tf.nn.relu(_pred_biochem), tf.nn.relu(_biochem_y))))# / config.BATCH_SIZE_BIOCHEM

            _weight_regularize = tf.multiply(tf.nn.l2_loss(_freeAGO_guide_offset) \
                                + tf.nn.l2_loss(_w1) \
                                + tf.nn.l2_loss(_w2) \
                                # + tf.nn.l2_loss(_w2_1) \
                                + tf.nn.l2_loss(_w3) \
                                + tf.nn.l2_loss(_w4), config.LAMBDA)

            # _weight_regularize = tf.multiply(tf.nn.l2_loss(_w1) \
            #                     + tf.nn.l2_loss(_w2) \
            #                     # + tf.nn.l2_loss(_w2_1) \
            #                     + tf.nn.l2_loss(_w3) \
            #                     + tf.nn.l2_loss(_w4), config.LAMBDA)


            # _pred_logfc_normed = _pred_logfc - tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1,1])
            # _repression_y_normed = _repression_y - tf.reshape(tf.reduce_mean(_repression_y, axis=1), [-1,1])
            # _repression_loss = (_repression_weight * tf.nn.l2_loss(tf.subtract(_pred_logfc_normed, _repression_y_normed))) / config.BATCH_SIZE_REPRESSION

            _pred_logfc_mean = tf.reshape(tf.reduce_mean(_pred_logfc, axis=1), [-1,1])
            _pred_logfc_max = tf.reshape(tf.reduce_max(_pred_logfc, axis=1), [-1,1])
            _pred_logfc_normed = _pred_logfc - _pred_logfc_max

            if config.BASELINE_METHOD is None:
                _repression_y_mean = tf.reshape(tf.reduce_mean(_repression_y, axis=1), [-1,1])
                _repression_y_normed = _repression_y - _repression_y_mean - (_pred_logfc_max - _pred_logfc_mean)
                _repression_loss = (_repression_weight * tf.nn.l2_loss(tf.subtract(_pred_logfc_normed, _repression_y_normed))) / config.BATCH_SIZE_REPRESSION
            
            else:
                _repression_loss = (_repression_weight * tf.nn.l2_loss(tf.subtract(_pred_logfc, _repression_y))) / config.BATCH_SIZE_REPRESSION


            _loss = _biochem_loss + _repression_loss + _weight_regularize

            tvars = tf.trainable_variables()
            gvars = [var for var in tvars if '_toggle' not in var.name]

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(config.STARTING_LEARNING_RATE, global_step,
                                                       2500, 0.90, staircase=True)

            _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(_update_ops):
                _train_step = tf.train.AdamOptimizer(config.STARTING_LEARNING_RATE).minimize(_loss, var_list=gvars, global_step=global_step)
                _train_step_trainable = tf.train.AdamOptimizer(config.STARTING_LEARNING_RATE).minimize(_loss, global_step=global_step)

        saver = tf.train.Saver(max_to_keep=config.NUM_EPOCHS+1)

        init_params_test = [
                    -5.5,
                    np.array([[0.0]]),
                    np.array([[-2.0]]),
                    config.DECAY_INIT,
                    config.UTR_COEF_INIT
                ]
        with tf.name_scope('test'):
            # split data into biochem and repression
            _repression_max_size_test = tf.placeholder(tf.int32, shape=[], name='repression_max_size')
            _repression_split_sizes_test = tf.placeholder(tf.int32, shape=[config.BATCH_SIZE_REPRESSION_TEST*2], name='repression_split_sizes')
            results_test = tf_helpers.both_steps_simple('test', 1, 0, config.BATCH_SIZE_REPRESSION_TEST, _pred_ind_values,
                                _repression_max_size_test, _repression_split_sizes_test, _utr_len, init_params_test)
            _, _, _freeAGO_mean_test, _freeAGO_guide_offset_test, _freeAGO_pass_offset_test, _freeAGO_all_test, _decay_test, _utr_coef_test, _pred_logfc_test = results_test


        saver = tf.train.Saver(max_to_keep=config.NUM_EPOCHS+1)

        ### TRAIN MODEL ###

        sess.run(tf.global_variables_initializer())

        if options.PRETRAIN != 'none':
            latest = tf.train.latest_checkpoint(options.PRETRAIN)
            print('Restoring from {}'.format(latest))
            saver.restore(sess, latest)

            # reset later layers
            # sess.run(_w2.initializer)
            # sess.run(_b2.initializer)
            # sess.run(_w4.initializer)
            # sess.run(_b4.initializer)

        # if options.RELOAD is not None:
        #     latest = tf.train.latest_checkpoint(options.RELOAD)
        #     print('Restoring from {}'.format(latest))
        #     saver.restore(sess, latest)

        # plot weights
        conv_weights = sess.run(_w1)
        xlabels = ['U','A','G','C']
        ylabels = ['A','U','C','G']
        helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(options.LOGDIR, 'convolution1_init.pdf'))

        # conv_weights = np.abs(sess.run(_w2))
        # conv_weights = np.sum(conv_weights, axis=(2,3))
        # vmin, vmax = np.min(conv_weights), np.max(conv_weights)
        # xlabels = ['s1', 's2']
        # ylabels = ['m2', 'm1']
        # fig = plt.figure(figsize=(4,4))
        # sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
        #             cmap=plt.cm.bwr, vmin=vmin, vmax=vmax)
        # plt.savefig(os.path.join(options.LOGDIR, 'convolution2_init.pdf'))
        # plt.close()

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

        print("Started training...")

        SWITCH_TRAINABLES = True

        step_list = []
        train_losses = []
        test_losses = []
        r2s = []
        last_batch = False
        prev_freeago = config.FREEAGO_INIT

        step = -1
        current_epoch = 0

        # save initial model
        times, times2 = [], []
        decay_list, utr_coef_list = [], []

        while True:

            t0 = time.time()

            # get repression data batch
            batch_genes, next_epoch, all_seqs, train_sizes, max_sites, batch_repression_y = repression_train_data.get_next_batch(config.BATCH_SIZE_REPRESSION, train_mirs)

            if next_epoch:
                current_epoch += 1

            # if none of the genes have sites, continue
            if max_sites == 0:
                continue

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

            if config.BATCH_SIZE_BIOCHEM > 0:
                # get biochem data batch
                _, biochem_train_batch = biochem_train_data.get_next_batch(config.BATCH_SIZE_BIOCHEM - 2)

                batch_biochem_y = []
                # fill in features for biochem data
                for mir, seq, logka in zip(biochem_train_batch['mir'], biochem_train_batch['seq'], biochem_train_batch['log ka']):
                    mirseq_one_hot = config.ONE_HOT_DICT[mir]

                    # add actual sequence
                    temp = np.outer(mirseq_one_hot, helpers.one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS))
                    batch_combined_x[current_ix, :, :] = temp
                    batch_biochem_y.append([logka])
                    current_ix += 1

                # add nonmatching sequence for miRNA
                seq = helpers.get_target_no_match(config.MIRSEQ_DICT[mir], config.SEQLEN)
                temp = np.outer(mirseq_one_hot, helpers.one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS))
                batch_combined_x[current_ix, :, :] = temp
                batch_biochem_y.append([0])
                current_ix += 1

                # add nonmatching sequence for a random miRNA sequence
                random_mir = helpers.generate_random_seq(config.MIRLEN)
                mirseq_one_hot = helpers.one_hot_encode(random_mir[::-1], config.MIR_NT_DICT, config.TARGETS)
                seq = helpers.get_target_no_match(random_mir, config.SEQLEN)
                temp = np.outer(mirseq_one_hot, helpers.one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS))
                batch_combined_x[current_ix, :, :] = temp
                batch_biochem_y.append([0])
                current_ix += 1
                
                batch_biochem_y = np.array(batch_biochem_y)
            else:
                batch_biochem_y = np.array([[0]])

            assert(current_ix == batch_combined_x.shape[0])
            batch_combined_x = np.expand_dims((batch_combined_x*4) - 0.25, 3)
            

            # run train step

            if config.BASELINE_METHOD is None:
                y_vals = batch_repression_y
            else:
                y_vals = batch_repression_y - train_tpm.loc[batch_genes][[config.BASELINE_METHOD]].values

            # make feed dict for training
            feed_dict = {
                    _keep_prob: config.KEEP_PROB_TRAIN,
                    _phase_train: True,
                    _repression_weight: config.REPRESSION_WEIGHT,
                    _combined_x: batch_combined_x,
                    _biochem_y: batch_biochem_y,
                    _repression_max_size: max_sites,
                    _repression_split_sizes: train_sizes,
                    _repression_y: y_vals,
                    _utr_len: tpm.loc[batch_genes][['utr_length']].values
                }

            times.append(time.time() - t0)
            t0 = time.time()

            if SWITCH_TRAINABLES:
                _, l1, l2, l3, train_loss = sess.run([_train_step_trainable, _biochem_loss, _repression_loss,
                                                          _weight_regularize, _loss], feed_dict=feed_dict)
            else:
                _, l1, l2, l3, train_loss = sess.run([_train_step, _biochem_loss, _repression_loss,
                                                      _weight_regularize, _loss], feed_dict=feed_dict)

            step += 1

            times2.append(time.time() - t0)

            # if (step % config.REPORT_INT) == 0:
            if next_epoch or (step == 0):

                print(np.mean(times), np.mean(times2))

                feed_dict = {
                    _keep_prob: 1.0,
                    _phase_train: False,
                    _repression_weight: config.REPRESSION_WEIGHT,
                    _combined_x: batch_combined_x,
                    _biochem_y: batch_biochem_y,
                    _repression_max_size: max_sites,
                    _repression_split_sizes: train_sizes,
                    _repression_y: y_vals,
                    _utr_len: tpm.loc[batch_genes][['utr_length']].values
                }

                if next_epoch:

                    step_list.append(current_epoch)
                    # step_list.append(step)

                    saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=current_epoch)
                    
                    # calculate and plot train performance
                    train_losses.append(train_loss)
                    print('Epoch {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(current_epoch, train_loss, l1, l2, l3))

                    fig = plt.figure(figsize=(7,5))
                    plt.plot(step_list, train_losses)
                    plt.savefig(os.path.join(options.LOGDIR, 'train_losses.png'))
                    plt.close()

                if config.BATCH_SIZE_BIOCHEM > 0:
                    train_biochem_preds = sess.run(_pred_biochem, feed_dict=feed_dict)

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(train_biochem_preds.flatten(), batch_biochem_y.flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'train_biochem_scatter.png'))
                    plt.close()

                train_repression_kds = sess.run(_pred_repression_flat, feed_dict=feed_dict)

                fig = plt.figure(figsize=(7,7))
                plt.hist(train_repression_kds.flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'train_repression_kds_hist.png'))
                plt.close()

                if config.BASELINE_METHOD is None:
                    train_repression_preds, train_repression_ys = sess.run([_pred_logfc_normed, _repression_y_normed],
                                                                            feed_dict=feed_dict)
                else:
                    train_repression_preds, train_repression_ys = sess.run([_pred_logfc, _repression_y],
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

                feed_dict = {
                    _keep_prob: 1.0,
                    _phase_train: False,
                    _combined_x: test_kds_combined_x
                }

                pred_test_kds = sess.run(_pred_ind_values, feed_dict=feed_dict)

                fig = plt.figure(figsize=(7,7))
                plt.scatter(pred_test_kds, test_kds_labels, s=20, c=test_kds_colors)
                plt.title(TEST_MIRNA)
                plt.savefig(os.path.join(options.LOGDIR, 'test_kd_scatter.png'))
                plt.close()

                fig = plt.figure(figsize=(7,7))
                plt.scatter(np.maximum(0,pred_test_kds), np.maximum(0,test_kds_labels), s=20, c=test_kds_colors)
                plt.title(TEST_MIRNA)
                plt.savefig(os.path.join(options.LOGDIR, 'test_kd_scatter_bounded.png'))
                plt.close()

                feed_dict = {
                    _keep_prob: 1.0,
                    _phase_train: False,
                    _combined_x: test_kds_combined_x2
                }

                pred_test_kds2 = sess.run(_pred_ind_values, feed_dict=feed_dict)

                fig = plt.figure(figsize=(7,7))
                plt.scatter(pred_test_kds2, test_kds_labels2, s=20, c=test_kds_colors2)
                plt.title(test_mir2)
                plt.savefig(os.path.join(options.LOGDIR, 'test_kd_scatter2.png'))
                plt.close()

                # if SWITCH_TRAINABLES == False:
                #     freeAGO_mean = sess.run(_freeAGO_mean)
                #     if next_epoch and (freeAGO_mean > prev_freeago):
                #         print('SWITCHED')
                #         SWITCH_TRAINABLES = True
                #     else:
                #         prev_freeago = freeAGO_mean

                current_decay = sess.run(_decay)
                current_freeAGO = sess.run(_freeAGO_all)
                current_utr_coef = sess.run(_utr_coef)
                current_freeAGO_mean = sess.run(_freeAGO_mean)
                current_freeAgo_guide_offset = np.mean(sess.run(_freeAGO_guide_offset))
                current_freeAgo_pass_offset = np.mean(sess.run(_freeAGO_pass_offset))

                decay_list.append(current_decay)
                utr_coef_list.append(current_utr_coef)

                fig = plt.figure(figsize=(10,5))
                plt.plot(decay_list)
                plt.savefig(os.path.join(options.LOGDIR, 'decay_fits.png'))
                plt.close()

                fig = plt.figure(figsize=(10,5))
                plt.plot(utr_coef_list)
                plt.savefig(os.path.join(options.LOGDIR, 'utr_coef_fits.png'))
                plt.close()

                print(current_freeAGO.reshape([NUM_TRAIN, 2]))
                print(current_decay)
                print(current_utr_coef)

                # check test miRNA
                assign_op1 = tf.assign(_freeAGO_mean_test, current_freeAGO_mean)
                assign_op2 = tf.assign(_freeAGO_guide_offset_test, np.array([[current_freeAgo_guide_offset]]))
                assign_op3 = tf.assign(_freeAGO_pass_offset_test, np.array([[current_freeAgo_pass_offset]]))
                assign_op4 = tf.assign(_decay_test, current_decay)
                assign_op5 = tf.assign(_utr_coef_test, current_utr_coef)
                sess.run([assign_op1, assign_op2, assign_op3, assign_op4, assign_op5])
                test_preds = []
                test_kd_vals = []
                test_genes = []

                blah = []
                for _ in range(43):

                    # get repression data batch
                    batch_genes, next_epoch, all_seqs, train_sizes, max_sites, batch_repression_y = repression_test_data.get_next_batch_no_shuffle(config.BATCH_SIZE_REPRESSION_TEST, test_mirs)
                    assert(next_epoch == False)

                    num_total_train_seqs = np.sum(train_sizes)
                    batch_combined_x = np.zeros([num_total_train_seqs, 4*config.MIRLEN, 4*config.SEQLEN])

                    # fill features for utr sites for both the guide and passenger strands
                    current_ix = 0
                    mirlist = test_mirs*config.BATCH_SIZE_REPRESSION_TEST
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

                    assert(current_ix == batch_combined_x.shape[0])
                    batch_combined_x = np.expand_dims((batch_combined_x*4) - 0.25, 3)

                    # make feed dict for training
                    feed_dict = {
                            _keep_prob: 1.0,
                            _phase_train: False,
                            _combined_x: batch_combined_x,
                            _repression_max_size_test: max_sites,
                            _repression_split_sizes_test: train_sizes,
                            _utr_len: tpm.loc[batch_genes][['utr_length']].values
                        }

                    test_preds += list(sess.run(_pred_logfc_test, feed_dict=feed_dict).flatten())
                    test_kd_vals += list(sess.run(_pred_ind_values, feed_dict=feed_dict).flatten())
                    test_genes += batch_genes

                    blah += batch_repression_y.flatten().tolist()

                fig = plt.figure(figsize=(7,7))
                plt.hist(test_kd_vals)
                plt.savefig(os.path.join(options.LOGDIR, 'test_kds_hist.png'))
                plt.close()

                assert(test_genes == list(test_tpm.index))
                repression_test_data.current_ix = 0

                if config.BASELINE_METHOD is None:
                    logfc_df = test_tpm[[TEST_MIRNA, 'nosite3', 'num_canon']]
                else:
                    logfc_df = test_tpm[[TEST_MIRNA, config.BASELINE_METHOD, 'num_canon']]


                print(np.sum(np.abs(logfc_df[TEST_MIRNA].values - np.array(blah))))

                logfc_df['pred'] = test_preds
                logfc_df = logfc_df.dropna()
                if config.BASELINE_METHOD is None:
                    logfc_df['actual'] = logfc_df[TEST_MIRNA] - logfc_df['nosite3']
                else:
                    logfc_df['actual'] = logfc_df[TEST_MIRNA] - logfc_df[config.BASELINE_METHOD]
                xs, ys = logfc_df['pred'].values, logfc_df['actual'].values
                r2 = stats.linregress(xs, ys)[2]**2
                r2s.append(r2)

                fig = plt.figure(figsize=(7,7))
                plt.scatter(xs, ys, s=20, c=-1*logfc_df['num_canon'], vmax=0, vmin=-8, cmap=plt.cm.plasma)
                plt.title('{:.3f}, {:.3f}, {:.3f}'.format(r2, current_freeAgo_guide_offset, current_freeAgo_pass_offset))
                plt.savefig(os.path.join(options.LOGDIR, 'test_logfc_scatter.png'))
                plt.close()

                fig = plt.figure()
                plt.plot(r2s)
                plt.savefig(os.path.join(options.LOGDIR, 'r_squared_testing.png'))
                plt.close()
                    
                # if last epoch, quit and write params to a file
                if current_epoch == config.NUM_EPOCHS:
                    current_freeAGO = current_freeAGO.reshape([NUM_TRAIN, 2])
                    freeAGO_df = pd.DataFrame({'mir': train_mirs,
                                               'guide': current_freeAGO[:, 0],
                                               'passenger': current_freeAGO[:, 1]})

                    freeAGO_df.to_csv(os.path.join(options.LOGDIR, 'freeAGO_final.txt'), sep='\t', index=False)
                    with open(os.path.join(options.LOGDIR, 'fitted_params.txt'), 'w') as outfile:
                        outfile.write('decay\t{}\n'.format(current_decay))
                        outfile.write('utr_coef\t{}\n'.format(current_utr_coef))
                    break
