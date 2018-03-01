from optparse import OptionParser
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

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

    (options, args) = parser.parse_args()

    MIRLEN = 12
    SEQLEN = 12
    BATCH_SIZE_BIOCHEM = 100
    BATCH_SIZE_REPRESSION = 2
    KEEP_PROB_TRAIN = 0.5
    STARTING_LEARNING_RATE = 0.002
    LAMBDA = 0.001
    NUM_EPOCHS = 100
    REPORT_INT = 50
    REPRESSION_WEIGHT = 1.0
    ZERO_OFFSET = 0.0
    NORM_RATIO = 4.0

    HIDDEN1 = 4
    HIDDEN2 = 8
    HIDDEN3 = 16


    # make dictionary of reverse miRNA sequences trimmed to MIRLEN
    MIRSEQ_DICT_MIRLEN = {x: y[:MIRLEN][::-1] for (x,y) in helpers.MIRSEQ_DICT.items()}

    ONE_HOT_DICT = {x: helpers.one_hot_encode_nt_new(y, np.array(['A','T','C','G'])) for (x,y) in MIRSEQ_DICT_MIRLEN.items()}

    ### READ EXPRESSION DATA ###
    tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)

    MIRS = [x for x in tpm.columns if ('mir' in x) or ('lsy' in x)]
    assert options.TEST_MIRNA in MIRS

    # split miRNAs into training and testing
    train_mirs = [m for m in MIRS if m != options.TEST_MIRNA]
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

    print(data['mir'].unique())
    data = data[data['mir'] != 'mir7']
    print(data['mir'].unique())

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
    baseline_init = pd.read_csv(options.BASELINE_FILE, sep='\t', index_col=0)
    assert (len(baseline_init) == len(tpm))
    NUM_GENES = len(baseline_init)
    baseline_init = baseline_init.loc[tpm.index]['nosite_tpm'].values.reshape([NUM_GENES, 1])

    train_tpm[train_mirs] = train_tpm[train_mirs].values - baseline_init
    test_tpm[test_mirs] = test_tpm[test_mirs].values - baseline_init

    # make data objects for repression training data
    repression_train_data = objects.RepressionData(train_tpm)
    # repression_train_data.shuffle()

    repression_train_new = objects.RepressionDataNew(train_tpm)
    repression_train_new.get_seqs(train_mirs)

    # test on a subset of the test data to speed up testing
    # subset = np.random.choice(np.arange(len(test_tpm)), size=500)
    # test_tpm = test_tpm.iloc[subset]
    # test_logfc_labels = test_tpm[test_mirs].values

    # test_mirseq = MIRSEQ_DICT_MIRLEN[options.TEST_MIRNA]
    # test_seqs = []
    # test_site = helpers.SITE_DICT[options.TEST_MIRNA]
    # num_total_test_seqs = 0
    # for utr in test_tpm['Sequence']:
    #     seqs = helpers.get_seqs(utr, test_site, only_canon=False)
    #     test_seqs.append(seqs)
    #     num_total_test_seqs += len(seqs)

    # test_combined_x = np.zeros([num_total_test_seqs, 4*MIRLEN, 4*SEQLEN])
    # test_seq_utr_boundaries = [0]
    # current_ix = 0
    # for seq_list in test_seqs:

    #     if len(seq_list) == 0:
    #         test_seq_utr_boundaries.append(current_ix)

    #     else:
    #         for seq in seq_list:
    #             test_combined_x[current_ix, :, :] = helpers.make_square(test_mirseq, seq)
    #             current_ix += 1

    #         test_seq_utr_boundaries.append(current_ix)
    
    # test_combined_x = np.expand_dims(test_combined_x, 3)

    # get biochem data batch
    _, biochem_train_batch = biochem_train_data.get_next_batch(BATCH_SIZE_BIOCHEM)

    for _ in range(1):

        # OLD WAY
        ##################################
        T0 = time.time()
        t0 = time.time()

        # get repression data batch
        next_epoch, repression_train_batch = repression_train_data.get_next_batch(BATCH_SIZE_REPRESSION)

        all_seqs = []
        num_sites = 0
        for repression_row in repression_train_batch.iterrows():
            utr = repression_row[1]['Sequence']
            gene_seqs = []
            for mir in train_mirs:

                seqs = helpers.get_seqs(utr, helpers.SITE_DICT[mir], only_canon=False)
                # if current_epoch == 1:
                #     seqs = helpers.get_seqs(utr, helpers.SITE_DICT[mir], only_canon=True)
                # else:
                #     seqs = helpers.get_seqs(utr, helpers.SITE_DICT[mir], only_canon=False)
                gene_seqs.append(seqs)
                len_temp = len(seqs)

                if len_temp > num_sites:
                    num_sites = len_temp
            all_seqs.append(gene_seqs)

        if num_sites == 0:
            continue

        print('get seqs 1: {:.3}'.format(time.time() - t0))
        t0 = time.time()

        # print(all_seqs)

        batch_combined_x = np.zeros([(BATCH_SIZE_REPRESSION * NUM_TRAIN * num_sites) + BATCH_SIZE_BIOCHEM, 4*MIRLEN, 4*SEQLEN])
        batch_repression_mask = np.zeros([BATCH_SIZE_REPRESSION, NUM_TRAIN, num_sites])

        for counter1, big_seq_list in enumerate(all_seqs):

            for counter2, (mir, seq_list) in enumerate(zip(train_mirs, big_seq_list)):

                if len(seq_list) == 0:
                    continue

                mirseq = MIRSEQ_DICT_MIRLEN[mir]
                current = (counter1 * NUM_TRAIN * num_sites) + (counter2 * num_sites)
                for seq in seq_list:
                    batch_combined_x[current, :, :] = helpers.make_square(mirseq, seq)
                    current += 1
                batch_repression_mask[counter1, counter2, :len(seq_list)] = 1.0

        print('fill array 1: {:.3}'.format(time.time() - t0))
        t0 = time.time()

        batch_repression_y_old = repression_train_batch[train_mirs].values

        current = BATCH_SIZE_REPRESSION * NUM_TRAIN * num_sites
        for mirseq, seq in zip(biochem_train_batch['mirseq'], biochem_train_batch['seq']):
            batch_combined_x[current, :, :] = helpers.make_square(mirseq, seq)
            current += 1

        batch_combined_x_old = np.expand_dims(batch_combined_x, 3)
        batch_biochem_y = biochem_train_batch[['log ka']].values

        # print('OLD WAY: {:.3}'.format(time.time() - t0))
        print('finish array 1 1: {:.3}'.format(time.time() - t0))
        print('TOTAL 1: {:.3}'.format(time.time() - T0))

        all_seqs_old = all_seqs

        # FIND SEQS AHEAD OF TIME
        #####################################

        T0 = time.time()
        t0 = time.time()
        SEQ_NTS = np.array(['T','A','G','C'])
        # get repression data batch
        next_epoch, all_seqs, num_sites, batch_repression_y = repression_train_new.get_next_batch(BATCH_SIZE_REPRESSION, train_mirs)
        print('get seqs 2: {:.3}'.format(time.time() - t0))

        batch_combined_x = np.zeros([(BATCH_SIZE_REPRESSION * NUM_TRAIN * num_sites) + BATCH_SIZE_BIOCHEM, 4*MIRLEN, 4*SEQLEN])
        batch_repression_mask = np.zeros([BATCH_SIZE_REPRESSION, NUM_TRAIN, num_sites])
        for counter1, big_seq_list in enumerate(all_seqs):

            for counter2, (mir, seq_list) in enumerate(zip(train_mirs, big_seq_list)):

                if len(seq_list) == 0:
                    continue

                mirseq_one_hot = ONE_HOT_DICT[mir]
                current = (counter1 * NUM_TRAIN * num_sites) + (counter2 * num_sites)
                for seq in seq_list:
                    temp = np.outer(mirseq_one_hot, helpers.one_hot_encode_nt_new(seq, SEQ_NTS))
                    batch_combined_x[current, :, :] = temp - 0.25
                    current += 1
                batch_repression_mask[counter1, counter2, :len(seq_list)] = 1.0

        print('fill array 2: {:.3}'.format(time.time() - t0))
        t0 = time.time()

        current = BATCH_SIZE_REPRESSION * NUM_TRAIN * num_sites
        for mir, seq in zip(biochem_train_batch['mir'], biochem_train_batch['seq']):
            mirseq_one_hot = ONE_HOT_DICT[mir]
            temp = np.outer(mirseq_one_hot, helpers.one_hot_encode_nt_new(seq, SEQ_NTS))
            batch_combined_x[current, :, :] = temp - 0.25
            current += 1

        batch_combined_x = np.expand_dims(batch_combined_x, 3)
        batch_biochem_y = biochem_train_batch[['log ka']].values

        print('finish array 2: {:.3}'.format(time.time() - t0))
        print('TOTAL 2: {:.3}'.format(time.time() - T0))

        print(np.sum(np.abs(batch_combined_x_old - batch_combined_x)))
        print(np.sum(np.abs(batch_repression_y_old - batch_repression_y)))


        #         