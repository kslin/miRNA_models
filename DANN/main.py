from optparse import OptionParser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tensorflow as tf

import helpers
from model import inference
import train_model

def train_hyperparam(kd_data, logfc_data, adv_data, params, inference_func, logdir):

    test_set = kd_data.iloc[:params['TEST_SIZE']]
    train_set = kd_data.iloc[params['TEST_SIZE']:]

    test_set_logfc = logfc_data[~logfc_data['mir'].isin(['let7','lsy6'])]

    # learning_rate_l2 = 10**(np.random.random()*1.5 - 4)
    # learning_rate_disc = 10**(np.random.random()*1 - 2)
    # reg = 10**(np.random.random()*3 - 2)
    learning_rate_l2 = 0.001
    learning_rate_disc = 0.02
    # # reg = 10**(-1.4)
    reg = 0.5
    params['STARTING_LEARNING_RATE_L2'] = learning_rate_l2
    params['STARTING_LEARNING_RATE_DISC'] = learning_rate_disc
    params['LAMBDA'] = reg
    loss, logfc_loss, epoch = train_model.train_model(train_set, test_set, test_set_logfc, adv_data,
                                                          inference_func, params, logdir)

    return reg, learning_rate_l2, learning_rate_disc, logfc_loss, loss, epoch


def train_kd_only(train_mirs, kd_data, logfc_data, params, inference_func, logdir):

    print(params)

    test_set = kd_data.iloc[:params['TEST_SIZE']]
    train_set = kd_data.iloc[params['TEST_SIZE']:]

    test_set_logfc = logfc_data[~logfc_data['mir'].isin(['let7','lsy6'])]

    loss, logfc_loss, epoch = train_model.train_model(train_mirs, train_set, test_set, test_set_logfc,
                                                              inference_func, params, logdir)

    print('logfc_loss: {}, loss: {}\n, epoch:{}'.format(logfc_loss[-1], loss[-1], epoch))


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-i", "--infile", dest="INFILE", help="training data")
    parser.add_option("-f", "--logfc_file", dest="LOGFC_FILE", help="logfc data")
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs", default=None)

    (options, args) = parser.parse_args()

    params = {
                'MIRLEN': 20,
                'SEQLEN': 12,
                'IN_NODES': 1,
                'OUT_NODES': 1,
                'HIDDEN1': 2,
                'HIDDEN2': 8,
                'HIDDEN3': 32,
                'DISC1': 32,
                'ERROR_MODEL': 'l2',
                'MAX_EPOCH': 50,
                'BATCH_SIZE': 100,
                'ADV_BATCH_SIZE': 100,
                'REPORT_INT': 50,
                'KEEP_PROB_TRAIN': 0.5,
                'TEST_SIZE': 5000,
                'RESTORE_FROM': None,
                'LOG_SCALE': False,
                'NCOLS': 1
        }

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    metafile = open(os.path.join(options.LOGDIR, 'params.txt'), 'w')
    for key in sorted(params.keys()):
        metafile.write('{}: {}\n'.format(key, params[key]))

    # read in kd data:
    data = pd.read_csv(options.INFILE, header=None)
    data.columns = ['mirseq','seq','kd']
    data['sitem8'] = [helpers.complementary(mirseq[-8:-1]) for mirseq in data['mirseq']]
    data['color'] = [helpers.get_color(sitem8, seq) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]
    data['color2'] = [helpers.get_color(sitem8, seq[2:10]) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

    # get rid of sequences with sites out of register
    print(len(data))
    data = data[data['color'] == data['color2']].drop('color2',1)
    print(len(data))

    print(np.min(data['kd']))
    print(np.max(data['kd']))

    # log-transformation and zero-centering
    data['log_kd'] = (-1*np.log2(data['kd']) - 4)/6.4

    data = data.iloc[np.random.permutation(len(data))]

    print(data.head())

    train_sites = list(data['sitem8'].unique())

    # train_mirs = list(data['mirseq'].unique())
    # for mir in train_mirs:
    #     data[mir] = (data['mirseq'] == mir).astype(int)

    # params['NUM_CLASSES'] = len(train_mirs)

    # read in logfc data
    logfc = pd.read_csv(options.LOGFC_FILE, sep='\t')
    logfc['sitem8'] = [helpers.complementary(mirseq[-8:-1]) for mirseq in logfc['mirseq']]
    logfc['color'] = [helpers.get_color(sitem8, seq) for (sitem8, seq) in zip(logfc['sitem8'], logfc['seq'])]
    logfc['logFC'] /= -1
    # logfc = logfc[logfc['site_type'] != 'no site']
    logfc = logfc[logfc['color'] != 'grey']
    subset = np.random.choice(np.arange(len(logfc)), size=3000, replace=False)
    logfc = logfc.iloc[subset]

    print('len logfc')
    print(len(logfc))

    params['NUM_CLASSES'] = 2

    # create data for adversarial training
    adv = data[['mirseq','seq','color']]
    adv['keep'] = [int(np.random.random() > 0.9) if x == 'grey' else 1 for x in adv['color']]
    adv = adv[adv['keep'] == 1].drop('keep',1)
    print(adv['mirseq'].unique())

    num_grey = int(list(adv['color']).count('grey')/5)
    num_entries = int(len(adv)/5)
    num_blue = num_entries - num_grey

    print(num_grey, num_blue, num_entries)

    train_mirs = list(data['mirseq'].unique())
    temp = logfc[~logfc['mir'].isin(['mir1','mir124','mir155','let7','lsy6'])]
    test_mirs = temp['mir'].unique()
    test_mirseqs = [temp[temp['mir'] == m].iloc[0]['mirseq'] for m in test_mirs]
    test_mirseqs = [x[-20:] if len(x) >= 20 else ''.join(['A']*len(x) - 20) + x for x in test_mirseqs]

    print(len(train_mirs), len(test_mirseqs))


    def generate_not_pairing(length, not_equal):
        while True:
            seq = helpers.generate_random_seq(length)
            if seq != not_equal:
                return seq
    
    # new_mirseq = test_mirseqs*num_entries
    # new_sitem8 = [helpers.complementary(x[-8:-1]) for x in test_mirseqs]*num_entries
    # new_site = [helpers.complementary(x[-5:-1]) for x in test_mirseqs]*num_grey + \
    #            [helpers.complementary(x[-7:-1]) for x in test_mirseqs]*num_blue
    # new_last_nt = [generate_not_pairing(2, x[1:3]) for x in new_sitem8[:len(test_mirseqs)*num_grey]] + ['']*(num_blue*len(test_mirseqs))
    # new_up = [helpers.generate_random_seq(3) for x in range(num_entries*len(test_mirseqs))]

    # new_down = [helpers.generate_random_seq(3) for x in range(num_entries*len(test_mirseqs))]

    # new_adv = pd.DataFrame({'mirseq': new_mirseq, 'site': new_site, 'last_nt': new_last_nt, 'sitem8': new_sitem8,
    #                         'up': new_up, 'down': new_down})
    # new_adv['seq'] = new_adv['up'] + new_adv['last_nt'] + new_adv['site'] + new_adv['down']
    # new_adv['color'] = [helpers.get_color(sitem8, seq[2:10]) for (sitem8, seq) in zip(new_adv['sitem8'], new_adv['seq'])]

    num_entries *= 15
    num_grey *= 15
    num_blue *= 15


    new_mirseq = [helpers.generate_random_seq(20) for _ in range(num_entries)]
    new_sitem8 = [helpers.complementary(x[-8:-1]) for x in new_mirseq]
    new_site = [helpers.complementary(x[-5:-1]) for x in new_mirseq[:num_grey]] + \
               [helpers.complementary(x[-7:-1]) for x in new_mirseq[num_grey:]]
    new_last_nt = [generate_not_pairing(2, x[1:3]) for x in new_sitem8[:num_grey]] + (['']*num_blue)
    new_up = [helpers.generate_random_seq(3) for x in range(num_entries)]

    new_down = [helpers.generate_random_seq(3) for x in range(num_entries)]

    new_adv = pd.DataFrame({'mirseq': new_mirseq, 'site': new_site, 'last_nt': new_last_nt, 'sitem8': new_sitem8,
                            'up': new_up, 'down': new_down})
    new_adv['seq'] = new_adv['up'] + new_adv['last_nt'] + new_adv['site'] + new_adv['down']
    new_adv['color'] = [helpers.get_color(sitem8, seq[2:10]) for (sitem8, seq) in zip(new_adv['sitem8'], new_adv['seq'])]
    new_adv['overlap'] = [x in train_sites for x in new_sitem8]
    new_adv = new_adv[new_adv['overlap'] == False]

    print(len(new_adv))
    # print(new_adv.head())
    print(list(new_adv['color']).count('grey'))

    # new_adv = new_adv.drop(['up','down','last_nt','site'],1)

    # print(new_adv.head())

    # adv = pd.concat([adv, new_adv])
    # print(len(adv))




    # train_kd_only(train_mirs, data, logfc, params, inference, options.LOGDIR)

    # hyperparam_df = pd.DataFrame(None)
        # new_dir = os.path.join(options.LOGDIR, str(i))
        # train_hyperparam(train_mirs, data, logfc, params, inference, new_dir)

    # with open(os.path.join(options.LOGDIR, 'hyperparams3.txt'), 'w') as outfile:
    #     outfile.write('reg\tlr_l2\tlr_disc\tlogfc_loss\tloss\tepoch\ttime_to_calc\n')
    #     for run_num in range(50):
    #         t0 = time.time()
    #         results = list(train_hyperparam(train_mirs, data, logfc, params, inference, None))
    #         time_passed = time.time() - t0

    #         results.append(time_passed)
    #         outfile.write('\t'.join([str(x) for x in results]))
    #         outfile.write('\n')

    # for run_num in range(50):
    results = list(train_hyperparam(data, logfc, new_adv, params, inference, os.path.join(options.LOGDIR, 'minimax')))
    print(results)
            



