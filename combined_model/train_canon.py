from optparse import OptionParser
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

    parser = OptionParser()
    parser.add_option("--tpm_file", dest="TPM_FILE", help="tpm data")
    parser.add_option("--feature_file", dest="FEATURE_FILE", help="feature data")
    parser.add_option("--model_type", dest="MODEL_TYPE", help="which model to run")

    (options, args) = parser.parse_args()

    tpms = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)
    tpms.index.name = 'transcript'

    features_canon = pd.read_csv(options.FEATURE_FILE, sep='\t')
    features_canon = features_canon.set_index(keys=['transcript', 'mir'])

    # bound log KA
    features_canon['log_KA'] = np.maximum(features_canon['log_KA'].values, 0)
    canon_stypes = ['8mer', '7mer-m8', '7mer-a1', '6mer', '6mer-m8', '6mer-a1']
    canon4_stypes = ['8mer', '7mer-m8', '7mer-a1', '6mer']

    # one-hot categorical variables
    for stype in canon_stypes:
        features_canon[stype] = [float(s == stype) for s in features_canon['stype']]

    for feat in ['siRNA_1', 'siRNA_8', 'site_8']:
        for nt in ['A', 'C', 'G']:
            features_canon['{}{}'.format(feat, nt)] = (features_canon[feat] == nt).astype(float)
    
    upper_bound_dict = {
        '8mer': -0.03,
        '7mer-m8': -0.02,
        '7mer-a1': -0.01,
        '6mer': 0.0,
        '6mer-m8': 0.0,
        '6mer-a1': 0.0,
        'no site': 0.0
    }
    features_canon['upper_bound'] = [upper_bound_dict[x] for x in features_canon['stype']]
    nsites_canon = utils.get_nsites(features_canon)
    
    features_canon4 = features_canon[features_canon['stype'].isin(canon4_stypes)]
    nsites_canon4 = utils.get_nsites(features_canon4)

    # preds = features_canon4.reset_index().groupby(['transcript','mir']).agg({'upper_bound': np.sum})
    # actuals = tpms.reset_index().melt(id_vars=['transcript'])
    # actuals.columns = ['transcript', 'mir', 'tpm']

    # all_transcripts = list(tpms.index)
    # all_preds = np.zeros([len(all_transcripts), 5])
    # for ix, trans in enumerate(all_transcripts):
    #     for iy, mir in enumerate(mirs5):
    #         try:
    #             all_preds[ix, iy] = preds.loc[(trans, mir)]
    #         except:
    #             continue

    # print(np.sum(all_preds))

    # print(utils.get_r2_unnormed(all_preds, tpms[mirs5].values))

    # different sets of miRNAs
    mirs16 = sorted(['mir1','mir124','mir155','mir7','lsy6','mir153','mir139','mir144','mir223','mir137',
                    'mir205','mir143','mir182','mir199a','mir204','mir216b'])
    mirs5 = sorted(['mir1','mir124','mir155','mir7','lsy6'])

    num_batches = 10

    # shuffle transcripts and subset into 10 batches
    np.random.seed(0)
    tpms = tpms.iloc[np.random.permutation(len(tpms))]
    tpms['batch'] = [ix % num_batches for ix in range(len(tpms))]

    # TS7
    print('TS7')
    tf.reset_default_graph()
    norm_vars = ['TA', 'SPS', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'Off6m', 'ORF_8m']
    cat_vars = ['siRNA_1{}'.format(nt) for nt in ['A', 'C', 'G']] + \
                ['siRNA_8{}'.format(nt) for nt in ['A', 'C', 'G']] + \
                ['site_8{}'.format(nt) for nt in ['A', 'C', 'G']]

    all_vars = norm_vars + cat_vars

    features_for_ts = [features_canon4[canon4_stypes]]
    new_norm_vars = []
    for stype in canon4_stypes:
        new_norm_vars += [x + '_' + stype for x in norm_vars]
        temp = features_canon4[all_vars]
        stype_filter = features_canon4[[stype]].values
        temp.columns = [x + '_' + stype for x in temp.columns]

        temp *= stype_filter
        features_for_ts.append(temp)

    features_for_ts = pd.concat(features_for_ts, axis=1, join='inner')

    for col in features_for_ts.columns:
        if np.std(features_for_ts[col].values) < 0.00001:
            features_for_ts = features_for_ts.drop(columns=[col])

    all_vars = list(features_for_ts.columns)
    norm_vars = [x for x in new_norm_vars if x in all_vars]

    features_for_ts = pd.concat([features_for_ts, features_canon4['upper_bound']], axis=1, join='inner')

    model = models.BoundedLinearModel(len(all_vars))
    train_r2s, val_r2s = models.cross_val(tpms, features_for_ts, nsites_canon4, mirs16, mirs5, all_vars + ['upper_bound'], norm_vars, model, 2000, one_site=True)

    print(np.mean(train_r2s), np.std(train_r2s))
    print(np.mean(val_r2s), np.std(val_r2s))

    # TS7, multisite
    print('TS7, multisite')
    tf.reset_default_graph()
    model = models.BoundedLinearModel(len(all_vars))
    train_r2s, val_r2s = models.cross_val(tpms, features_for_ts, nsites_canon4, mirs16, mirs5, all_vars + ['upper_bound'], norm_vars, model, 2000, one_site=False)

    print(np.mean(train_r2s), np.std(train_r2s))
    print(np.mean(val_r2s), np.std(val_r2s))

    # TS7, multisite, 5 mirs
    tf.reset_default_graph()
    print('TS7, multisite, 5 mirs')
    features_for_ts = features_for_ts.query('mir in @mirs5')
    nsites_canon4 = nsites_canon4.query('mir in @mirs5')
    
    model = models.BoundedLinearModel(len(all_vars))
    train_r2s, val_r2s = models.cross_val(tpms, features_for_ts, nsites_canon4, mirs5, mirs5, all_vars + ['upper_bound'], norm_vars, model, 2000, one_site=False)

    print(np.mean(train_r2s), np.std(train_r2s))
    print(np.mean(val_r2s), np.std(val_r2s))


    # TS7, multisite, 5 mirs, with log KA
    print('TS7, multisite, 5 mirs, with log KA')
    tf.reset_default_graph()
    features_for_ts = pd.concat([features_for_ts, features_canon4['log_KA']], axis=1, join='inner')
    all_vars.append('log_KA')
    
    model = models.BoundedLinearModel(len(all_vars))
    train_r2s, val_r2s = models.cross_val(tpms, features_for_ts, nsites_canon4, mirs5, mirs5, all_vars + ['upper_bound'], norm_vars, model, 2000, one_site=False)

    print(np.mean(train_r2s), np.std(train_r2s))
    print(np.mean(val_r2s), np.std(val_r2s))

    # TS7, multisite, 5 mirs, with log KA, - redundant
    print('TS7, multisite, 5 mirs, with log KA, - redundant')
    tf.reset_default_graph()
    norm_vars = ['log_KA', 'TA', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'Off6m', 'ORF_8m']
    features_for_ts = pd.concat([features_canon4, features_canon4['upper_bound']], axis=1, join='inner')
    features_for_ts = features_for_ts.query('mir in @mirs5')

    model = models.BoundedLinearModel(len(all_vars))
    train_r2s, val_r2s = models.cross_val(tpms, features_for_ts, nsites_canon4, mirs5, mirs5, all_vars + ['upper_bound'], norm_vars, model, 2000, one_site=False)

    print(np.mean(train_r2s), np.std(train_r2s))
    print(np.mean(val_r2s), np.std(val_r2s))

    # TS7, multisite, 6 canon, 5mirs
    print('TS7, multisite, 6 canon, 5mirs')
    tf.reset_default_graph()
    model = models.BoundedLinearModel(len(all_vars))
    train_r2s, val_r2s = models.cross_val(tpms, features_for_ts, nsites_canon, mirs5, mirs5, all_vars + ['upper_bound'], norm_vars, model, 2000, one_site=False)

    print(np.mean(train_r2s), np.std(train_r2s))
    print(np.mean(val_r2s), np.std(val_r2s))

    # # TS7, multisite, 6 canon
    # print('TS7, multisite, 6 canon')
    # tf.reset_default_graph()
    # norm_vars = ['TA', 'SPS', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
    # cat_vars = canon_stypes + \
    #             ['siRNA_1{}'.format(nt) for nt in ['A', 'C', 'G']] + \
    #             ['siRNA_8{}'.format(nt) for nt in ['A', 'C', 'G']] + \
    #             ['site_8{}'.format(nt) for nt in ['A', 'C', 'G']]

    # all_vars = norm_vars + cat_vars + ['upper_bound']
    # model = models.BoundedLinearModel(len(all_vars) - 1)
    # train_r2s, val_r2s = models.cross_val(tpms, features_canon, nsites_canon, mirs16, mirs5, all_vars, norm_vars, model, 200, one_site=False)

    # print(np.mean(train_r2s), np.std(train_r2s))
    # print(np.mean(val_r2s), np.std(val_r2s))

    # # TS7, multisite, 6 canon, 5 mirs
    # print('TS7, multisite, 6 canon, 5 mirs')
    # tf.reset_default_graph()
    # model = models.BoundedLinearModel(len(all_vars) - 1)
    # train_r2s, val_r2s = models.cross_val(tpms, features_canon, nsites_canon, mirs5, mirs5, all_vars, norm_vars, model, 200, one_site=False)

    # print(np.mean(train_r2s), np.std(train_r2s))
    # print(np.mean(val_r2s), np.std(val_r2s))

    # # TS7, multisite, 6 canon, 5 mirs, + logKA
    # print('TS7, multisite, 6 canon, 5 mirs, + logKA')
    # tf.reset_default_graph()
    # norm_vars = ['log_KA', 'TA', 'SPS', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
    # cat_vars = canon_stypes + \
    #             ['siRNA_1{}'.format(nt) for nt in ['A', 'C', 'G']] + \
    #             ['siRNA_8{}'.format(nt) for nt in ['A', 'C', 'G']] + \
    #             ['site_8{}'.format(nt) for nt in ['A', 'C', 'G']]

    # all_vars = norm_vars + cat_vars + ['upper_bound']

    # model = models.BoundedLinearModel(len(all_vars) - 1)
    # train_r2s, val_r2s = models.cross_val(tpms, features_canon, nsites_canon, mirs5, mirs5, all_vars, norm_vars, model, 200, one_site=False)

    # print(np.mean(train_r2s), np.std(train_r2s))
    # print(np.mean(val_r2s), np.std(val_r2s))

    # # TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant
    # print('TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant')
    # tf.reset_default_graph()
    # norm_vars = ['log_KA', 'TA', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
    # all_vars = norm_vars + canon_stypes + ['upper_bound']

    # model = models.BoundedLinearModel(len(all_vars) - 1)
    # train_r2s, val_r2s = models.cross_val(tpms, features_canon, nsites_canon, mirs5, mirs5, all_vars, norm_vars, model, 200, one_site=False)

    # print(np.mean(train_r2s), np.std(train_r2s))
    # print(np.mean(val_r2s), np.std(val_r2s))

    # # TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound
    # print('TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound')
    # tf.reset_default_graph()
    # norm_vars = ['Local_AU', 'Threep', 'SA', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
    # all_vars = ['log_KA'] + norm_vars + canon_stypes

    # model = models.SigmoidModel(1, len(all_vars) - 1, len(mirs5))
    # train_r2s, val_r2s = models.cross_val(tpms, features_canon, nsites_canon, mirs5, mirs5, all_vars, norm_vars, model, 200, one_site=False)

    # print(np.mean(train_r2s), np.std(train_r2s))
    # print(np.mean(val_r2s), np.std(val_r2s))

    # # TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound, include energetic features in sigmoid
    # print('TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound, include energetic features in sigmoid')
    # tf.reset_default_graph()
    # model = models.SigmoidModel(4, len(all_vars) - 4, len(mirs5))
    # train_r2s, val_r2s = models.cross_val(tpms, features_canon, nsites_canon, mirs5, mirs5, all_vars, norm_vars, model, 200, one_site=False)

    # print(np.mean(train_r2s), np.std(train_r2s))
    # print(np.mean(val_r2s), np.std(val_r2s))

    # # TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound, double sigmoid
    # print('TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound, double sigmoid')
    # tf.reset_default_graph()
    # model = models.DoubleSigmoidModel(1, len(norm_vars), len(mirs5))
    # train_r2s, val_r2s = models.cross_val(tpms, features_canon, nsites_canon, mirs5, mirs5, all_vars, norm_vars, model, 200, one_site=False)

    # print(np.mean(train_r2s), np.std(train_r2s))
    # print(np.mean(val_r2s), np.std(val_r2s))

    # # TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound, double sigmoid, include energetics
    # print('TS7, multisite, 6 canon, 5 mirs, + logKA, - redundant, with sigmoid, no bound, double sigmoid, include energetics')
    # tf.reset_default_graph()
    # model = models.DoubleSigmoidModel(4, len(all_vars) - 4, len(mirs5))
    # train_r2s, val_r2s = models.cross_val(tpms, features_canon, nsites_canon, mirs5, mirs5, all_vars, norm_vars, model, 200, one_site=False)

    # print(np.mean(train_r2s), np.std(train_r2s))
    # print(np.mean(val_r2s), np.std(val_r2s))

