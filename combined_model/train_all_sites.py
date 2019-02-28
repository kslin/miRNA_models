import os

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf

import models
import utils

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 100
tf.logging.set_verbosity(tf.logging.DEBUG)


if __name__ == '__main__':

    TPM_FILE = '/lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt'
    FEATURE_FILE = 'outputs/all_sites/features.txt'

    tpms = pd.read_csv(TPM_FILE, sep='\t', index_col=0)

    features = pd.read_csv(FEATURE_FILE, sep='\t',
        usecols=['transcript', 'mir', 'log_KA', 'Local_AU', 'Threep', 'SA', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT'])
    features = features.set_index(keys=['transcript', 'mir'])

    print(len(features))
    features = features[features['log_KA'] > 0]
    print(len(features))

    nsites = utils.get_nsites(features)

    mirs5 = sorted(['mir1','mir124','mir155','mir7','lsy6'])

    num_batches = 10

    # shuffle transcripts and subset into 10 batches
    np.random.seed(0)
    tpms = tpms.iloc[np.random.permutation(len(tpms))]
    tpms['batch'] = [ix % num_batches for ix in range(len(tpms))]


    # TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound, double sigmoid, all sites
    print('TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound, double sigmoid, all sites')
    tf.reset_default_graph()
    norm_vars = ['Local_AU', 'Threep', 'SA', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT']
    all_vars = ['log_KA'] + norm_vars
    model = models.DoubleSigmoidModel(1, len(norm_vars), len(mirs5))
    train_r2s, val_r2s = models.cross_val(tpms, features, nsites, mirs5, mirs5, all_vars, norm_vars, model, 200, one_site=False)

    print(np.mean(train_r2s), np.std(train_r2s))
    print(np.mean(val_r2s), np.std(val_r2s))

    # TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound, double sigmoid, include energetics
    print('TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound, double sigmoid, include energetics, all sites')
    tf.reset_default_graph()
    model = models.DoubleSigmoidModel(4, len(all_vars) - 4, len(mirs5))
    train_r2s, val_r2s = models.cross_val(tpms, features, nsites, mirs5, mirs5, all_vars, norm_vars, model, 200, one_site=False)

    print(np.mean(train_r2s), np.std(train_r2s))
    print(np.mean(val_r2s), np.std(val_r2s))

