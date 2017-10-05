from optparse import OptionParser
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tensorflow as tf

import helpers
from model import inference_2layer, inference_3layer
import train_model

def train_hyperparam(kd_data, logfc_data, params, inference_func):
    params['TRAIN_LOGFC'] = True
    params['TRAIN_KD'] = True

    test_set = kd_data.iloc[:params['TEST_SIZE']]
    train_set = kd_data.iloc[params['TEST_SIZE']:]

    all_mirs = logfc_data['mir'].unique()

    for i in range(params['NUM_RUNS']):
        test_mirs = np.random.choice([a for a in all_mirs if a not in ['mir1','mir155','mir124','lsy6']], size=3, replace=False)
        train_mirs = list(set(all_mirs) - set(test_mirs))
        print('\t'.join(test_mirs) + '\n')

        train_set_logfc = logfc_data[logfc_data['mir'].isin(train_mirs)]
        test_set_logfc = logfc_data[logfc_data['mir'].isin(test_mirs)]
        
        for j in range(10):
            learning_rate = 10**(np.random.random()*2 - 4)
            reg = 10**(np.random.random()*2 - 5)
            params['STARTING_LEARNING_RATE'] = learning_rate
            params['LAMBDA'] = reg
            loss, logfc_loss, epoch = train_model.train_model(train_set_logfc, test_set_logfc,
                                                              train_set, test_set,
                                                              inference_func, params)
            print('reg: {:.3}, lr: {:.3}, logfc_loss: {}, loss: {}\n, epoch:{}'.format(reg, learning_rate,
                                                                             logfc_loss[-1], loss[-1], epoch))


def train_kd_only(kd_data, logfc_data, params, inference_func, logdir):
    params['TRAIN_LOGFC'] = False
    params['TRAIN_KD'] = True

    print(params)

    test_set = kd_data.iloc[:params['TEST_SIZE']]
    train_set = kd_data.iloc[params['TEST_SIZE']:]

    all_mirs = logfc_data['mir'].unique()
    test_mirs = [a for a in all_mirs if a not in ['let7','lsy6']]
    train_set_logfc = None
    test_set_logfc = logfc_data[logfc_data['mir'].isin(test_mirs)]

    loss, logfc_loss, epoch = train_model.train_model(train_set_logfc, test_set_logfc,
                                                              train_set, test_set,
                                                              inference_func, params, logdir)

    print('logfc_loss: {}, loss: {}\n, epoch:{}'.format(logfc_loss[-1], loss[-1], epoch))


def train_logfc_only(kd_data, logfc_data, params, train_mirs, test_mirs, inference_func, logdir):
    params['TRAIN_LOGFC'] = True
    params['TRAIN_KD'] = False

    print(params)
    
    test_set = kd_data.iloc[:params['TEST_SIZE']]
    train_set = None

    train_set_logfc = logfc_data[logfc_data['mir'].isin(train_mirs)]
    test_set_logfc = logfc_data[logfc_data['mir'].isin(test_mirs)]

    loss, logfc_loss, epoch = train_model.train_model(train_set_logfc, test_set_logfc,
                                                              train_set, test_set,
                                                              inference_func, params, logdir)

    print('logfc_loss: {}, loss: {}\n, epoch:{}'.format(logfc_loss[-1], loss[-1], epoch))


def train_logfc_and_kd(kd_data, logfc_data, params, train_mirs, test_mirs, inference_func, logdir):
    params['TRAIN_LOGFC'] = True
    params['TRAIN_KD'] = True

    print(params)
    
    test_set = kd_data.iloc[:params['TEST_SIZE']]
    train_set = kd_data.iloc[params['TEST_SIZE']:]

    train_set_logfc = logfc_data[logfc_data['mir'].isin(train_mirs)]
    test_set_logfc = logfc_data[logfc_data['mir'].isin(test_mirs)]

    loss, logfc_loss, epoch = train_model.train_model(train_set_logfc, test_set_logfc,
                                                              train_set, test_set,
                                                              inference_func, params, logdir)

    print('logfc_loss: {}, loss: {}\n, epoch:{}'.format(logfc_loss[-1], loss[-1], epoch))


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-i", "--infile", dest="INFILE", help="training data")
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs", default=None)

    (options, args) = parser.parse_args()

    params = {
                'MIRLEN': 20,
                'SEQLEN': 12,
                'IN_NODES': 1,
                'OUT_NODES': 1,
                'HIDDEN1': 4,
                'HIDDEN2': 16,
                'HIDDEN3': 8,
                'ERROR_MODEL': 'l2',
                'MAX_EPOCH': 10,
                'BATCH_SIZE': 200,
                'LOGFC_BATCH_SIZE': 200,
                'REPORT_INT': 50,
                'KEEP_PROB_TRAIN': 0.5,
                'TEST_SIZE': 5000,
                'RESTORE_FROM': None,
                'NUM_RUNS': 2,
                'STARTING_LEARNING_RATE': 0.001,
                'STARTING_LEARNING_RATE_LOGFC': 0.0005,
                'LAMBDA': 0.00005,
                'LOG_SCALE': False,
                'NCOLS': 1
        }

    print(params)

    if not os.path.isdir(options.LOGDIR):
        os.makedirs(options.LOGDIR)

    metafile = open(os.path.join(options.LOGDIR, 'params.txt'), 'w')
    for key in sorted(params.keys()):
        metafile.write('{}: {}\n'.format(key, params[key]))

    # read in kd data:
    data = pd.read_csv(options.INFILE, header=None)
    data.columns = ['mirseq','seq','kd']
    data['sitem8'] = [helpers.complementaryT(mirseq[-8:-1]) for mirseq in data['mirseq']]
    data['color'] = [helpers.get_color_old(sitem8, seq) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

    # simple log-transformation and zero-centering
    data['log_kd'] = (-1*np.log2(data['kd']) - 3)/10

    shuffle_ix = np.random.permutation(len(data))
    data = data.iloc[shuffle_ix]

    # read in logfc data
    logfc = pd.read_csv('/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/for_nn.txt',sep='\t')
    logfc['sitem8'] = [helpers.complementaryT(mirseq[-8:-1]) for mirseq in logfc['mirseq']]
    logfc['color'] = [helpers.get_color_old(sitem8, seq) for (sitem8, seq) in zip(logfc['sitem8'], logfc['seq'])]
    logfc['logFC'] /= -3

    # train_hyperparam(data, logfc, params, inference_2layer)

    # train_kd_only(data, logfc, params, inference_2layer, os.path.join(options.LOGDIR, 'kd_only'))

    all_mirs = logfc['mir'].unique()
    train_mirs = [a for a in all_mirs if a not in ['let7']]
    test_mirs = [a for a in all_mirs if a not in ['let7']]

    train_logfc_and_kd(data, logfc, params, train_mirs, test_mirs, inference_2layer,
                             os.path.join(options.LOGDIR, 'logfc_and_kd'))

    # for i in range(len(all_mirs)):
    #     for j in range(i+1, len(all_mirs)):
    #         mir1 = all_mirs[i]
    #         mir2 = all_mirs[j]
    #         print(mir1,mir2)
    #         train_mirs = [a for a in all_mirs if a not in [mir1, mir2]]
    #         test_mirs = [mir1, mir2]
    #         mir_logdir = os.path.join(options.LOGDIR, mir1 + '_' + mir2)

    #         if not os.path.isdir(mir_logdir):
    #             os.makedirs(mir_logdir)

    #         # train_logfc_only(data, logfc, params, train_mirs, test_mirs, inference_2layer,
    #         #                  os.path.join(mir_logdir, 'logfc_only'))

    #         train_logfc_and_kd(data, logfc, params, train_mirs, test_mirs, inference_2layer,
    #                          os.path.join(mir_logdir, 'logfc_and_kd'))



