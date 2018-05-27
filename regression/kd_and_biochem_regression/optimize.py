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

    # np.random.seed(0)
    # seqs = [helpers.generate_random_seq(12) for _ in range(100)]
    # mirseqs = [helpers.generate_random_seq(12) for _ in range(100)]

    # t0 = time.time()
    # collection1 = []
    # for (seq, mirseq) in zip(seqs, mirseqs):
    #     one_hot_mirseq = helpers.one_hot_encode_nt_new(mirseq, config.MIR_NTS)
    #     collection1.append(one_hot_mirseq)

    # print(time.time() - t0)

    # nt_dict = {nt:ix for (ix, nt) in enumerate(config.MIR_NTS)}
    # blah = np.eye(4)
    # t0 = time.time()
    # collection2 = []
    # for (seq, mirseq) in zip(seqs, mirseqs):
    #     mirseq = [nt_dict[nt] for nt in mirseq]
    #     one_hot_mirseq = blah[mirseq].flatten() * 2
    #     collection2.append(one_hot_mirseq)

    # print(time.time() - t0)

    # for i in range(10):
    #     print(np.sum(np.abs(collection1[i] - collection2[i])))

    def one_hot_encode(seq, nt_dict, targets):
        seq = [nt_dict[nt] for nt in seq]
        return targets[seq].flatten()

    PRETRAIN_SAVE_PATH = os.path.join(options.LOGDIR, 'pretrain_saved')
    SAVE_PATH = os.path.join(options.LOGDIR, 'saved')

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    ### READ EXPRESSION DATA ###
    tpm = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0, nrows=500)

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
    
    data = data[~data['mir'].isin(test_mirs)]
    print(len(data))

    # create data object
    biochem_train_data = data_objects_endog.BiochemData(data)
    biochem_train_data.shuffle()

    # make data objects for repression training data
    repression_train_data = data_objects_endog.RepressionData(train_tpm)
    repression_train_data.shuffle()
    repression_train_data.get_seqs(train_mirs)

    ONE_HOT_DICT = {x: one_hot_encode(y, config.MIR_NT_DICT, config.TARGETS) for (x,y) in config.MIRSEQ_DICT_MIRLEN.items()}

    times = []

    for _ in range(2):

        # get repression data batch
        batch_genes, next_epoch, all_seqs, train_sizes, max_sites, batch_repression_y = repression_train_data.get_next_batch2(config.BATCH_SIZE_REPRESSION, train_mirs)

        # get biochem data batch
        _, biochem_train_batch = biochem_train_data.get_next_batch(config.BATCH_SIZE_BIOCHEM)

        t0 = time.time()

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

        assert(current_ix == batch_combined_x.shape[0])

        batch_combined_x_v1 = np.expand_dims(batch_combined_x, 3)
        batch_biochem_y = biochem_train_batch[['log ka']].values + config.ZERO_OFFSET # OFFSET

        print((time.time() - t0))

        t0 = time.time()

        num_total_train_seqs = np.sum(train_sizes)
        batch_combined_x = np.zeros([num_total_train_seqs + config.BATCH_SIZE_BIOCHEM, 4*config.MIRLEN, 4*config.SEQLEN])

        # fill features for utr sites for both the guide and passenger strands
        current_ix = 0
        mirlist = train_mirs*config.BATCH_SIZE_REPRESSION
        for mir, (seq_list_guide, seq_list_pass) in zip(mirlist, all_seqs):
            mirseq_one_hot_guide = ONE_HOT_DICT[mir]
            mirseq_one_hot_pass = ONE_HOT_DICT[mir + '*']

            for seq in seq_list_guide:
                if 'U' in seq:
                    print(seq)
                temp = np.outer(mirseq_one_hot_guide, one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS))
                batch_combined_x[current_ix, :, :] = temp
                current_ix += 1

            for seq in seq_list_pass:
                if 'U' in seq:
                    print(seq)
                temp = np.outer(mirseq_one_hot_pass, one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS))
                batch_combined_x[current_ix, :, :] = temp
                current_ix += 1

        # fill in features for biochem data
        for mir, seq in zip(biochem_train_batch['mir'], biochem_train_batch['seq']):
            mirseq_one_hot = ONE_HOT_DICT[mir]
            temp = np.outer(mirseq_one_hot, one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS))
            batch_combined_x[current_ix, :, :] = temp
            current_ix += 1

        assert(current_ix == batch_combined_x.shape[0])

        batch_combined_x_v2 = np.expand_dims((batch_combined_x*4) - 0.25, 3)
        batch_biochem_y = biochem_train_batch[['log ka']].values + config.ZERO_OFFSET # OFFSET

        print((time.time() - t0))
        print(np.sum(np.abs(batch_combined_x_v1 - batch_combined_x_v2)))
