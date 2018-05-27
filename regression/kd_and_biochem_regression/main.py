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

import config, helpers, data_objects, model_objects

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-k", "--kdfile", dest="KD_FILE", help="kd data")
    parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    parser.add_option("-i", "--init_baselines", dest="BASELINE_FILE", help="initial baseline data")
    parser.add_option("-m", "--mirna", dest="TEST_MIRNA", help="testing miRNA")
    parser.add_option("--hidden1", dest="HIDDEN1", type=int, help="number of nodes in layer 1")
    parser.add_option("--hidden2", dest="HIDDEN2", type=int, help="number of nodes in layer 2")
    parser.add_option("--hidden3", dest="HIDDEN3", type=int, help="number of nodes in layer 3")
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("-d", "--do_training", dest="DO_TRAINING", help="toggle training", action="store_true", default=False)
    parser.add_option("-p", "--pretrain", dest="PRETRAIN", help="directory with pretrained weights", default=None)

    (options, args) = parser.parse_args()

    PRETRAIN_SAVE_PATH = os.path.join(options.LOGDIR, 'pretrain_saved')
    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    ### READ EXPRESSION DATA ###
    tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)

    MIRS = [x for x in tpm.columns if ('mir' in x) or ('lsy' in x)]

    # split miRNAs into training and testing
    if options.TEST_MIRNA == 'none':
        train_mirs = MIRS
        test_mirs = ['mir139']
        TEST_MIRNA = 'mir139'
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

    # split tpm data into training and testing
    train_tpm = tpm[train_mirs + ['Sequence']]
    test_tpm = tpm[test_mirs + ['Sequence']]

    ### READ KD DATA ###
    data = pd.read_csv(options.KD_FILE, sep='\t')
    data.columns = ['mir','mirseq_full','seq','log kd','stype']
    data = data[~data['mir'].isin(test_mirs)]

    # zero-center and normalize Ka's
    data['log ka'] = ((-1.0 * data['log kd']) + config.ZERO_OFFSET)
    data['mirseq'] = [config.MIRSEQ_DICT_MIRLEN[mir] for mir in data['mir']]
    data['sitem8'] = [helpers.rev_comp(mirseq[1:8]) for mirseq in data['mirseq_full']]
    data['color'] = [helpers.get_color(sitem8, seq) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]
    data['color2'] = [helpers.get_color(sitem8, seq[2:10]) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

    # get rid of sequences with sites out of register
    print(len(data))
    data = data[data['color'] == data['color2']].drop('color2',1)
    print(len(data))

    # create data object
    biochem_train_data = data_objects.BiochemData(data, cutoff=0.95)
    biochem_train_data.shuffle()

    ### READ INITIAL BASELINE ###
    baseline_df = pd.read_csv(options.BASELINE_FILE, sep='\t', index_col=0)
    assert (len(baseline_df) == len(tpm))
    NUM_GENES = len(baseline_df)
    baseline_df = baseline_df.loc[tpm.index]
    baseline_original = baseline_df.copy()
    # gene_order = tpm.index

    # make data objects for repression training data
    repression_train_data = data_objects.RepressionDataNew_with_passenger(train_tpm)
    repression_train_data.shuffle()
    repression_train_data.get_seqs(train_mirs)

    # nt_dict = {'A':'T', 'T':'A', 'C': 'G', 'G':'C'}
    # site = config.SITE_DICT[options.TEST_MIRNA]
    # test_mirseq_one_hot_guide = config.ONE_HOT_DICT[options.TEST_MIRNA]
    # test_seqs = ['AA' + site + 'AAA',
    #              'AA' + site + 'TAA',
    #              'GG' + site + 'CCC',
    #              'AA' + nt_dict[site[0]] + site[1:] + 'TAA',
    #              'AA' + site[:3] + nt_dict[site[3]] + site[4:] + 'TAA']

    # site = config.SITE_DICT['let7']
    # train_mirseq_one_hot_guide = config.ONE_HOT_DICT['let7']
    # train_seqs = ['AA' + site + 'AAA',
    #              'AA' + site + 'TAA',
    #              'GG' + site + 'CCC',
    #              'AA' + nt_dict[site[0]] + site[1:] + 'TAA',
    #              'AA' + site[:3] + nt_dict[site[3]] + site[4:] + 'TAA']

    # test_combined_x = np.zeros([len(test_seqs)*2, 4*config.MIRLEN, 4*config.SEQLEN])
    # ix = 0
    # for i, seq in enumerate(test_seqs):
    #     temp = np.outer(test_mirseq_one_hot_guide, helpers.one_hot_encode_nt_new(seq, config.SEQ_NTS))
    #     test_combined_x[ix, :, :] = temp - 0.25
    #     ix += 1

    # for i, seq in enumerate(train_seqs):
    #     temp = np.outer(train_mirseq_one_hot_guide, helpers.one_hot_encode_nt_new(seq, config.SEQ_NTS))
    #     test_combined_x[ix, :, :] = temp - 0.25
    #     ix += 1

    # print(ix)
    # test_combined_x = np.expand_dims(test_combined_x, 3)

    # test on a subset of the test data to speed up testing
    subset = np.random.choice(np.arange(len(test_tpm)), size=400)
    test_tpm = test_tpm.iloc[subset]

    # get test miRNA labels
    test_tpm_labels = test_tpm[test_mirs].values

    # get test sequences for sense strand
    test_site_guide = config.SITE_DICT[TEST_MIRNA]
    test_site_pass = config.SITE_DICT[TEST_MIRNA + '*']
    test_seqs, test_seq_sizes = [], []
    for utr in test_tpm['Sequence']:

        # add seqs for guide strand
        seqs_guide = helpers.get_seqs(utr, test_site_guide, seqlen=config.SEQLEN)

        # add seqs for passenger strand
        seqs_pass = helpers.get_seqs(utr, test_site_pass, seqlen=config.SEQLEN)

        test_seqs.append((seqs_guide, seqs_pass))
        test_seq_sizes += [len(seqs_guide), len(seqs_pass)]

    # make empty array to fill with features
    num_total_test_seqs = np.sum(test_seq_sizes)
    test_combined_x = np.zeros([num_total_test_seqs, 4*config.MIRLEN, 4*config.SEQLEN])
    
    test_mirseq_one_hot_guide = config.ONE_HOT_DICT[TEST_MIRNA]
    test_mirseq_one_hot_pass = config.ONE_HOT_DICT[TEST_MIRNA + '*']

    # iterate through seqs and add to feature array
    current_ix = 0
    for seq_list_guide, seq_list_pass in test_seqs:

        for seq in seq_list_guide:
            temp = np.outer(test_mirseq_one_hot_guide, helpers.one_hot_encode_nt_new(seq, config.SEQ_NTS))
            test_combined_x[current_ix, :, :] = temp - 0.25
            current_ix += 1

        for seq in seq_list_pass:
            temp = np.outer(test_mirseq_one_hot_pass, helpers.one_hot_encode_nt_new(seq, config.SEQ_NTS))
            test_combined_x[current_ix, :, :] = temp - 0.25
            current_ix += 1
    
    test_combined_x = np.expand_dims(test_combined_x, 3)
    test_seq_sizes = np.array(test_seq_sizes)

    test_freeAGO_site = config.FREEAGO_SITE_DICT[TEST_MIRNA]
    test_freeAGO_pass = config.FREEAGO_PASS_DICT[TEST_MIRNA]
    test_freeAGO_diff = test_freeAGO_site - test_freeAGO_pass
    

    ### DEFINE MODEL ###

    # reset and build the neural network
    tf.reset_default_graph()

    # start session
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:
        
        # set up model
        ConvNet = model_objects.Model(config.MIRLEN, config.SEQLEN, NUM_TRAIN, options.LOGDIR, baseline_df, sess)
        ConvNet.build_model(options.HIDDEN1, options.HIDDEN2, options.HIDDEN3)
        # ConvNet.build_model_all_conv(options.HIDDEN1, options.HIDDEN2)

        if options.DO_TRAINING:
            freeAGO_init = []
            for _ in range(NUM_TRAIN):
                freeAGO_init += [-5, -7]

            freeAGO_init = np.array(freeAGO_init).reshape([1, NUM_TRAIN*2, 1])
            # print(freeAGO_init)
            ConvNet.add_repression_layers_mean_offset(config.BATCH_SIZE_REPRESSION, config.BATCH_SIZE_BIOCHEM,
                                                        freeAGO_init, config.NORM_RATIO)

        # initialize variables
        ConvNet.initialize_vars()

        ### PRETRAIN MODEL ###
        if options.PRETRAIN is not None:

            if options.PRETRAIN  == 'pretrain':

                print("Doing pre-training")

                # plot random initialized weights
                # ConvNet.plot_w1(os.path.join(options.LOGDIR, 'convolution1_start.pdf'))
                # ConvNet.plot_w3(os.path.join(options.LOGDIR, 'convolution3_start.pdf'))

                # pretrain on generated site-type-based data
                losses = []
                for pretrain_step in range(2000):
                    pretrain_batch_x, pretrain_batch_y = helpers.make_pretrain_data(100, config.MIRLEN, config.SEQLEN)
                    pretrain_batch_y = (pretrain_batch_y + config.ZERO_OFFSET) / config.NORM_RATIO

                    l = ConvNet.pretrain(pretrain_batch_x, pretrain_batch_y)
                    losses.append(l)

                    if (pretrain_step % 100) == 0:
                        print(pretrain_step)

                train_pred = ConvNet.predict_ind_values(pretrain_batch_x)

                fig = plt.figure(figsize=(7,7))
                plt.scatter(train_pred.flatten(), pretrain_batch_y.flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'pretrain_train_scatter.png'))
                plt.close()

                test_x, test_y = helpers.make_pretrain_data(100, config.MIRLEN, config.SEQLEN)
                test_y = (test_y + config.ZERO_OFFSET) / config.NORM_RATIO
                pred_pretrain = ConvNet.predict_ind_values(test_x)

                fig = plt.figure(figsize=(7,7))
                plt.scatter(pred_pretrain.flatten(), test_y.flatten())
                plt.savefig(os.path.join(options.LOGDIR, 'pretrain_test_scatter.png'))
                plt.close()

                fig = plt.figure(figsize=(7,5))
                plt.plot(losses)
                plt.savefig(os.path.join(options.LOGDIR, 'pretrain_losses.png'))
                plt.close()

                ConvNet.save_pretrained_weights(os.path.join(PRETRAIN_SAVE_PATH, 'model'))

                print("Finished pre-training")

            else:
                # restore previously pretrained weights
                ConvNet.restore_pretrained_weights(options.PRETRAIN)

            # plot weights after pre-training
            # ConvNet.plot_w1(os.path.join(options.LOGDIR, 'convolution1_pretrained.pdf'))
            # ConvNet.plot_w3(os.path.join(options.LOGDIR, 'convolution3_pretrained.pdf'))

        ### TRAIN MODEL ###

        if options.DO_TRAINING:

            print("Started training...")

            # reset later variables
            # ConvNet.reset_final_layer()

            # ConvNet.plot_w3(os.path.join(options.LOGDIR, 'convolution3_reset.pdf'))

            step_list = []
            train_losses = []
            test_losses = []
            last_batch = False

            step = -1
            current_epoch = 0

            # save initial model
            ConvNet.save(os.path.join(SAVE_PATH, 'model'), current_epoch)

            while True:

                # get repression data batch
                batch_genes, next_epoch, all_seqs, train_sizes, max_sites, batch_expression_y = repression_train_data.get_next_batch2(config.BATCH_SIZE_REPRESSION, train_mirs)

                if next_epoch:
                    current_epoch += 1
                    # config.REPRESSION_WEIGHT += 0.2
                    if repression_train_data.num_epochs >= config.NUM_EPOCHS:
                        last_batch = True

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
                        temp = np.outer(mirseq_one_hot_guide, helpers.one_hot_encode_nt_new(seq, config.SEQ_NTS))
                        batch_combined_x[current_ix, :, :] = temp - 0.25
                        current_ix += 1

                    for seq in seq_list_pass:
                        temp = np.outer(mirseq_one_hot_pass, helpers.one_hot_encode_nt_new(seq, config.SEQ_NTS))
                        batch_combined_x[current_ix, :, :] = temp - 0.25
                        current_ix += 1

                # fill in features for biochem data
                for mir, seq in zip(biochem_train_batch['mir'], biochem_train_batch['seq']):
                    mirseq_one_hot = config.ONE_HOT_DICT[mir]
                    temp = np.outer(mirseq_one_hot, helpers.one_hot_encode_nt_new(seq, config.SEQ_NTS))
                    batch_combined_x[current_ix, :, :] = temp - 0.25
                    current_ix += 1

                assert (current_ix == batch_combined_x.shape[0])

                batch_combined_x = np.expand_dims(batch_combined_x, 3)
                batch_biochem_y = biochem_train_batch[['log ka']].values

                # run train step
                train_loss, b_loss, weight_reg, r_loss = ConvNet.train(batch_genes, batch_combined_x, batch_biochem_y, max_sites, train_sizes, batch_expression_y)

                step += 1

                # if (step % config.REPORT_INT) == 0:

                    # feed_dict = {
                    #         ConvNet._keep_prob: config.KEEP_PROB_TRAIN,
                    #         ConvNet._phase_train: False,
                    #         ConvNet._repression_weight: config.REPRESSION_WEIGHT,
                    #         ConvNet._combined_x: batch_combined_x,
                    #         ConvNet._biochem_y: batch_biochem_y,
                    #         ConvNet._repression_max_size: max_sites,
                    #         ConvNet._repression_split_sizes: train_sizes,
                    #         ConvNet._expression_y: batch_expression_y}

                    # blah1, blah2 = sess.run([ConvNet._pred_tpm, ConvNet._real_tpm], feed_dict=feed_dict)

                    # print(np.min(blah1), np.mean(blah1), np.max(blah1))
                    # print(np.min(blah2), np.mean(blah2), np.max(blah2))

                    # print(train_loss, b_loss, r_loss)

                # print(train_loss, b_loss, r_loss)
                # if step == 5:
                #     next_epoch = True
                if next_epoch:
                    print(ConvNet.FIT_INTERCEPT)
                    if current_epoch >= config.SWITCH_EPOCH:
                        ConvNet.FIT_INTERCEPT = True

                    # save model
                    ConvNet.save(os.path.join(SAVE_PATH, 'model'), current_epoch)

                    print(step, train_loss, b_loss, weight_reg, r_loss)
                    
                    # calculate and plot train performance
                    step_list.append(current_epoch)
                    # step_list.append(step)
                    train_losses.append(train_loss)

                    fig = plt.figure(figsize=(7,5))
                    plt.plot(step_list, train_losses)
                    plt.savefig(os.path.join(options.LOGDIR, 'train_losses.png'))
                    plt.close()

                    train_biochem_preds = ConvNet.predict_ind_values(batch_combined_x)[-1 * config.BATCH_SIZE_BIOCHEM:, :]

                    intercept = ConvNet.BASELINES.loc[batch_genes][['nosite_tpm']].values

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(train_biochem_preds.flatten(), batch_biochem_y.flatten())
                    plt.savefig(os.path.join(options.LOGDIR, 'train_biochem_scatter.png'))
                    plt.close()

                    train_repression_preds = ConvNet.predict_repression(batch_combined_x, max_sites, train_sizes)

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(train_repression_preds, batch_expression_y - intercept)
                    plt.savefig(os.path.join(options.LOGDIR, 'train_repression_scatter.png'))
                    plt.close()

                    # plot weights
                    # ConvNet.plot_w1(os.path.join(options.LOGDIR, 'convolution1.pdf'))
                    # ConvNet.plot_w3(os.path.join(options.LOGDIR, 'convolution3.pdf'))

                    current_freeAGO = np.mean(sess.run(ConvNet._freeAGO_all).reshape([NUM_TRAIN, 2])[:, 0])
                    print('current freeAGO mean: {:.3}'.format(current_freeAGO))
                    # current_slope = sess.run(ConvNet._slope)
                    # print('current slope: {:.3}'.format(current_slope))
                    current_decay = sess.run(ConvNet._decay)
                    print('current decay: {:.3}'.format(current_decay))


                    # predict nbound for test miRNA
                    feed_dict = {
                                    ConvNet._keep_prob: 1.0,
                                    ConvNet._phase_train: False,
                                    ConvNet._combined_x: test_combined_x
                                }



                    pred_ind_values_test = ConvNet.predict_ind_values(test_combined_x).flatten()

                    nbound_guide, nbound_pass = [], []
                    prev = 0
                    guide = True
                    for size in test_seq_sizes:
                        temp = pred_ind_values_test[prev: prev+size]
                        prev += size
                        if guide:
                            ind_nbound = helpers.sigmoid(temp + current_freeAGO)
                            nbound_guide.append(np.sum(ind_nbound))
                            guide = False
                        else:
                            ind_nbound = helpers.sigmoid(temp + (current_freeAGO - test_freeAGO_diff))
                            nbound_pass.append(np.sum(ind_nbound))
                            guide = True

                    pred_nbound_test = np.array(nbound_guide) + np.array(nbound_pass)

                    pred_logfc_test = -1 * np.log1p(pred_nbound_test / np.exp(current_decay))
                    # pred_logfc_test = current_slope * pred_nbound_test
                    baselines = ConvNet.BASELINES.loc[test_tpm.index]['nosite_tpm'].values

                    test_logfc_labels = test_tpm_labels.flatten() - baselines

                    # calculate and plot test performance
                    test_losses.append(np.sum((pred_logfc_test - test_logfc_labels)**2)/len(test_tpm))

                    fig = plt.figure(figsize=(7,5))
                    plt.plot(step_list, test_losses)
                    plt.savefig(os.path.join(options.LOGDIR, 'test_losses.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7,7))
                    plt.scatter(pred_logfc_test, test_logfc_labels, s=30)
                    rsq = helpers.calc_rsq(pred_logfc_test, test_logfc_labels)
                    rsq2 = stats.linregress(pred_logfc_test, test_logfc_labels)[2]**2
                    plt.title('R2 = {:.3}, {:.3}'.format(rsq, rsq2))
                    plt.savefig(os.path.join(options.LOGDIR, 'test_scatter.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7,7))
                    plt.hist(pred_ind_values_test.flatten(), bins=100)
                    plt.savefig(os.path.join(options.LOGDIR, 'test_biochem_hist.png'))
                    plt.close()

                    fig = plt.figure(figsize=(7,7))
                    plt.hist(baseline_df['nosite_tpm'], bins=100)
                    plt.savefig(os.path.join(options.LOGDIR, 'nosite_tpm_hist.png'))
                    plt.close()

                    if last_batch:
                        # print(stats.linregress(pred_nbound_test.flatten(), test_logfc_labels.flatten()))
                        print('Repression epochs: {}'.format(repression_train_data.num_epochs))
                        print('Biochem epochs: {}'.format(biochem_train_data.num_epochs))
                        # print('Global slope: {:.3}'.format(sess.run(ConvNet._slope)))
                        print('Global decay rate: {:.3}'.format(sess.run(ConvNet._decay)))
                        trained_freeAGO = sess.run(ConvNet._freeAGO_all).flatten().reshape([NUM_TRAIN, 2])
                        print('Fitted free AGO:')
                        for m, f in zip(train_mirs, trained_freeAGO):
                            print('{}: {:.3}, {:.3}'.format(m, f[0], f[1]))

                        freeAGO_df = pd.DataFrame({'mir': train_mirs,
                                                   'guide': trained_freeAGO[:, 0],
                                                   'passenger': trained_freeAGO[:, 1]})

                        freeAGO_df.to_csv(os.path.join(options.LOGDIR, 'freeAGO_final.txt'), sep='\t', index=False)
                        baseline_df.to_csv(os.path.join(options.LOGDIR, 'final_baselines.txt'), sep='\t')

                        fig = plt.figure(figsize=(7,7))
                        plt.scatter(baseline_df['nosite_tpm'], baseline_original['nosite_tpm'])
                        plt.xlabel('new')
                        plt.ylabel('original')
                        plt.savefig(os.path.join(options.LOGDIR, 'nosite_scatter.png'))
                        plt.close()

                        break
