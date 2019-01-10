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

    # parser = OptionParser()
    # parser.add_option("-k", "--kdfile", dest="KD_FILE", help="kd data")
    # parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    # parser.add_option("-i", "--init_baselines", dest="BASELINE_FILE", help="initial baseline data")
    # parser.add_option("--hidden1", dest="HIDDEN1", type=int, help="number of nodes in layer 1")
    # parser.add_option("--hidden2", dest="HIDDEN2", type=int, help="number of nodes in layer 2")
    # parser.add_option("--hidden3", dest="HIDDEN3", type=int, help="number of nodes in layer 3")
    # parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs")
    # parser.add_option("-d", "--do_training", dest="DO_TRAINING", help="toggle training", action="store_true", default=False)
    # parser.add_option("-p", "--pretrain", dest="PRETRAIN", help="directory with pretrained weights", default=None)

    # (options, args) = parser.parse_args()

    # PRETRAIN_SAVE_PATH = os.path.join(options.LOGDIR, 'pretrain_saved')
    # SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    # MIRLEN, SEQLEN = 12, 12
    # BATCH_SIZE_REPRESSION, BATCH_SIZE_BIOCHEM = 3, 3

    # if not os.path.isdir(options.LOGDIR):
    #     os.makedirs(options.LOGDIR)

    # ### READ EXPRESSION DATA ###
    # tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0, nrows=10)

    # train_mirs = ['mir1','mir124']
    # test_mirs = ['mir155']

    # NUM_TRAIN = len(train_mirs)
    # NUM_TEST = len(test_mirs)

    # print(NUM_TRAIN, NUM_TEST)

    # # split tpm data into training and testing
    # train_tpm = tpm[train_mirs + ['Sequence']]
    # test_tpm = tpm[test_mirs + ['Sequence']]

    # ### READ KD DATA ###
    # data = pd.read_csv(options.KD_FILE, sep='\t', nrows=10)
    # data.columns = ['mir','mirseq_full','seq','log kd','stype']
    # # data = data[~data['mir'].isin(test_mirs)]
    # print(data.head())

    # # convert to Ka's
    # data['log ka'] = (-1.0 * data['log kd'])
    # data['mirseq'] = [config.MIRSEQ_DICT_MIRLEN[mir] for mir in data['mir']]
    # data['sitem8'] = [helpers.rev_comp(mirseq[1:8]) for mirseq in data['mirseq_full']]
    # data['color'] = [helpers.get_color(sitem8, seq) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]
    # data['color2'] = [helpers.get_color(sitem8, seq[2:10]) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

    # # get rid of sequences with sites out of register
    # print(len(data))
    # data = data[data['color'] == data['color2']].drop('color2',1)
    # print(len(data))

    # # create data object
    # biochem_train_data = data_objects.BiochemData(data, cutoff=0.95)
    # # biochem_train_data.shuffle()

    # ### READ INITIAL BASELINE ###
    # baseline_df = pd.read_csv(options.BASELINE_FILE, sep='\t', index_col=0, nrows=10)
    # assert (len(baseline_df) == len(tpm))
    # NUM_GENES = len(baseline_df)
    # baseline_df = baseline_df.loc[tpm.index]
    # baseline_original = baseline_df.copy()
    # # gene_order = tpm.index

    # # make data objects for repression training data
    # repression_train_data = data_objects.RepressionDataNew_with_passenger(train_tpm)
    # # repression_train_data.shuffle()
    # repression_train_data.get_seqs(train_mirs)

    BATCH_SIZE_REPRESSION, BATCH_SIZE_BIOCHEM = 3,1
    NUM_TRAIN = 2
    

     # reset and build the neural network
    tf.reset_default_graph()

    # start session
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:
        
        # set up model
        ConvNet = model_objects.Model(config.MIRLEN, config.SEQLEN, NUM_TRAIN, '', None, sess)
        ConvNet.build_model(4, 8, 16)


        freeAGO_init = []
        for _ in range(NUM_TRAIN):
            freeAGO_init += [-5, -7]

        freeAGO_init = np.array(freeAGO_init).reshape([1, NUM_TRAIN*2, 1])
        ConvNet.add_repression_layers_mean_offset(BATCH_SIZE_REPRESSION, BATCH_SIZE_BIOCHEM,
                                                    freeAGO_init, 4)

        # initialize variables
        ConvNet.initialize_vars()

        MAX_SITES = 3
        train_sizes = np.array([1,2, 2,3,
                                0,1, 1,1,
                                1,0, 0,0])

        total_num_seqs = np.sum(train_sizes) + BATCH_SIZE_BIOCHEM
        pred_ind_values = [5, 1.5, 7,
                           2, 8, 1, 2, 3,
                           9,
                           4, 3,
                           2,
                           6]
        pred_ind_values = np.array(pred_ind_values).reshape([total_num_seqs, 1]) / 4


        feed_dict = {
                            ConvNet._keep_prob: 1.0,
                            ConvNet._phase_train: False,
                            ConvNet._pred_ind_values: pred_ind_values,
                            ConvNet._repression_weight: config.REPRESSION_WEIGHT,
                            ConvNet._repression_max_size: MAX_SITES,
                            ConvNet._repression_split_sizes: train_sizes
                        }

        print(sess.run([ConvNet._pred_nbound_split, ConvNet._pred_nbound, ConvNet._pred_logfc, ConvNet._pred_tpm], feed_dict=feed_dict))

        # # get repression data batch
        # batch_genes, next_epoch, all_seqs, train_sizes, max_sites, batch_expression_y = repression_train_data.get_next_batch2(BATCH_SIZE_REPRESSION, train_mirs)

        # # get biochem data batch
        # # _, biochem_train_batch = biochem_train_data.get_next_batch(BATCH_SIZE_BIOCHEM)

        # num_total_train_seqs = np.sum(train_sizes)
        # batch_combined_x = np.zeros([num_total_train_seqs + BATCH_SIZE_BIOCHEM, 4*MIRLEN, 4*SEQLEN])

        # # fill features for utr sites for both the guide and passenger strands
        # current_ix = 0
        # mirlist = train_mirs*BATCH_SIZE_REPRESSION
        # for mir, (seq_list_guide, seq_list_pass) in zip(mirlist, all_seqs):
        #     mirseq_one_hot_guide = config.ONE_HOT_DICT[mir]
        #     mirseq_one_hot_pass = config.ONE_HOT_DICT[mir + '*']

        #     for seq in seq_list_guide:
        #         temp = np.outer(mirseq_one_hot_guide, helpers.one_hot_encode_nt_new(seq, config.SEQ_NTS))
        #         batch_combined_x[current_ix, :, :] = temp - 0.25
        #         current_ix += 1

        #     for seq in seq_list_pass:
        #         temp = np.outer(mirseq_one_hot_pass, helpers.one_hot_encode_nt_new(seq, config.SEQ_NTS))
        #         batch_combined_x[current_ix, :, :] = temp - 0.25
        #         current_ix += 1

        # # fill in features for biochem data
        # mirseq_one_hot = config.ONE_HOT_DICT['let7']
        # for seq in ['AACTACCTCAAA', 'GGCTACCTCACC', 'GGCTACCTCATT']:
        #     temp = np.outer(mirseq_one_hot, helpers.one_hot_encode_nt_new(seq, config.SEQ_NTS))
        #     batch_combined_x[current_ix, :, :] = temp - 0.25
        #     current_ix += 1

        # assert (current_ix == batch_combined_x.shape[0])

        # # print(batch_combined_x)
        # print(train_sizes)
        # print(batch_combined_x.shape)

        # batch_combined_x = np.expand_dims(batch_combined_x, 3)
        # # batch_biochem_y = biochem_train_batch[['log ka']].values

        # feed_dict = {
        #             model._keep_prob: 1.0,
        #             model._phase_train: False,
        #             model._combined_x: batch_combined_x,
        #             model._repression_weight: 1.0,
        #             model._repression_max_size: np.max(train_sizes),
        #             model._repression_split_sizes: train_sizes,
        #             model._expression_y: batch_expression_y 
        #             # model._intercept: baseline_df.loc[batch_genes][['nosite_tpm']].values
        #             }

        # results1 = sess.run(model._pred_ind_values, feed_dict=feed_dict).flatten()
        # print(results1.shape)
        # print(results1)
        # results2 = sess.run(model._pred_repression, feed_dict=feed_dict)
        # # print(results2)
        # print(results2.shape)
        # prev = 0
        # ix = 0
        # for batch in range(BATCH_SIZE_REPRESSION):
        #     for j in range(NUM_TRAIN*2):
        #         total = 0
        #         for blah in results2[batch, j]:
        #             if blah != 0:
        #                 total += 1.0 / (1.0 + np.exp(6 - blah))

        #         print(total)
        # results = sess.run(model._pred_nbound_split, feed_dict=feed_dict)
        # print(results)
        # results = sess.run(model._pred_nbound, feed_dict=feed_dict)
        # print(results)
        # print(-1*np.log1p(results))

        # results = sess.run(model._pred_logfc, feed_dict=feed_dict)
        # print(results)

        # print('intercept fit')
        # results = sess.run(model._intercept_fit, feed_dict=feed_dict)
        # print(results)

        # print('pred tpm')
        # results = sess.run(model._pred_tpm_fit, feed_dict=feed_dict)
        # print(results)

        # print('repression loss weights')
        # results = sess.run(model._repression_loss_weights, feed_dict=feed_dict)
        # print(results)

        # print(np.mean(batch_expression_y, axis=1))
        # # print(train_tpm.head())
        # # print(baseline_df.head())


