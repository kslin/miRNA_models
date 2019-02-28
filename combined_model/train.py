from optparse import OptionParser
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tensorflow as tf

import models
import utils

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 100
tf.logging.set_verbosity(tf.logging.DEBUG)


def expand_feats_stypes(features, stypes, expand_vars, single_vars):
    expanded_features = []
    for stype in stypes:
        temp = features[expand_vars]
        stype_filter = features[[stype]].values
        temp.columns = [x + ':' + stype for x in temp.columns]
        temp *= stype_filter
        expanded_features.append(temp)

    expanded_features.append(features[single_vars])
    expanded_features = pd.concat(expanded_features, axis=1, join='inner')

    # get rid of columns of all zeros, for example 6mer PCT
    for col in expanded_features.columns:
        if np.std(expanded_features[col].values) < 0.00001:
            expanded_features = expanded_features.drop(columns=[col])

    return expanded_features


def one_hot_features(features, stypes):

    # one-hot categorical variables
    for stype in stypes:
        features[stype] = [float(s == stype) for s in features['stype']]

    cat_vars = []
    for feat in ['siRNA_1', 'siRNA_8', 'site_8']:
        for nt in ['A', 'C', 'G']:
            features['{}{}'.format(feat, nt)] = (features[feat] == nt).astype(float)
            cat_vars.append(feat + nt)

    return features, cat_vars


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_file", dest="TPM_FILE", help="tpm data")
    parser.add_option("--feature_file", dest="FEATURE_FILE", help="feature data")
    parser.add_option("--model_type", dest="MODEL_TYPE", help="which model to run")
    parser.add_option("--out_folder", dest="OUT_FOLDER", help="folder for outputs", default=None)

    (options, args) = parser.parse_args()

    tpms = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)
    tpms.index.name = 'transcript'

    # shuffle transcripts and subset into 10 batches
    num_batches = 10
    np.random.seed(0)
    tpms = tpms.iloc[np.random.permutation(len(tpms))]
    tpms['batch'] = [ix % num_batches for ix in range(len(tpms))]
    test_tpms = tpms[tpms['batch'] == 0]
    tpms = tpms[tpms['batch'] != 0]

    features = pd.read_csv(options.FEATURE_FILE, sep='\t')
    features = features.set_index(keys=['transcript', 'mir'])

    upper_bound_dict = {
        '8mer': -0.03,
        '7mer-m8': -0.02,
        '7mer-a1': -0.01,
        '6mer': 0.0,
        '6mer-m8': 0.0,
        '6mer-a1': 0.0,
        'no site': 0.0
    }

    mirs16 = sorted(['mir1','mir124','mir155','mir7','lsy6','mir153','mir139','mir144','mir223','mir137',
                    'mir205','mir143','mir182','mir199a','mir204','mir216b'])
    mirs5 = sorted(['mir1','mir124','mir155','mir7','lsy6'])

    canon4_stypes = ['8mer', '7mer-m8', '7mer-a1', '6mer']
    canon6_stypes = ['8mer', '7mer-m8', '7mer-a1', '6mer', '6mer-m8', '6mer-a1']

    if options.MODEL_TYPE == 'TS7':
        # get data
        features = features[features['stype'].isin(canon4_stypes)]
        nsites = utils.get_nsites(features)
        features['upper_bound'] = [upper_bound_dict[x] for x in features['stype']]
        features, cat_vars = one_hot_features(features, canon4_stypes)

        # define vars
        single_vars = canon4_stypes + ['upper_bound']
        norm_vars = ['TA', 'SPS', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'Off6m', 'ORF_8m']
        features = expand_feats_stypes(features, canon4_stypes, norm_vars + cat_vars, single_vars)
        all_vars = list(features.columns)
        norm_vars = [x for x in all_vars if x.split(':')[0] in norm_vars]

        print(features.head())
        print(all_vars)
        sys.exit()

        # train model
        model = models.BoundedLinearModel(len(all_vars) - 1)
        one_site, train_mirs, val_mirs = True, mirs16, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite':
        # get data
        features = features[features['stype'].isin(canon4_stypes)]
        nsites = utils.get_nsites(features)
        features['upper_bound'] = [upper_bound_dict[x] for x in features['stype']]
        features, cat_vars = one_hot_features(features, canon4_stypes)

        # define vars
        single_vars = canon4_stypes + ['upper_bound']
        norm_vars = ['TA', 'SPS', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'Off6m', 'ORF_8m']
        features = expand_feats_stypes(features, canon4_stypes, norm_vars + cat_vars, single_vars)
        all_vars = list(features.columns)
        norm_vars = [x for x in all_vars if x.split(':')[0] in norm_vars]

        # train model
        model = models.BoundedLinearModel(len(all_vars) - 1)
        one_site, train_mirs, val_mirs = False, mirs16, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs':
        # get data
        features = features[features['stype'].isin(canon4_stypes)]
        features = features.query('mir in @mirs5')
        nsites = utils.get_nsites(features)
        features['upper_bound'] = [upper_bound_dict[x] for x in features['stype']]
        features, cat_vars = one_hot_features(features, canon4_stypes)

        # define vars
        single_vars = canon4_stypes + ['upper_bound']
        norm_vars = ['TA', 'SPS', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'Off6m', 'ORF_8m']
        features = expand_feats_stypes(features, canon4_stypes, norm_vars + cat_vars, single_vars)
        all_vars = list(features.columns)
        norm_vars = [x for x in all_vars if x.split(':')[0] in norm_vars]

        # train model
        model = models.BoundedLinearModel(len(all_vars) - 1)
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA':
        # get data
        features = features[features['stype'].isin(canon4_stypes)]
        features = features.query('mir in @mirs5')
        nsites = utils.get_nsites(features)
        features['upper_bound'] = [upper_bound_dict[x] for x in features['stype']]
        features, cat_vars = one_hot_features(features, canon4_stypes)

        # define vars
        single_vars = canon4_stypes + ['upper_bound']
        norm_vars = ['log_KA', 'TA', 'SPS', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'Off6m', 'ORF_8m']
        features = expand_feats_stypes(features, canon4_stypes, norm_vars + cat_vars, single_vars)
        all_vars = list(features.columns)
        norm_vars = [x for x in all_vars if x.split(':')[0] in norm_vars]

        # train model
        model = models.BoundedLinearModel(len(all_vars) - 1)
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA_noredundant':
        # get data
        features = features[features['stype'].isin(canon4_stypes)]
        features = features.query('mir in @mirs5')
        nsites = utils.get_nsites(features)
        features['upper_bound'] = [upper_bound_dict[x] for x in features['stype']]
        features, cat_vars = one_hot_features(features, canon4_stypes)
        cat_vars = canon4_stypes

        # define vars
        single_vars = canon4_stypes + ['upper_bound']
        norm_vars = ['log_KA', 'TA', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'Off6m', 'ORF_8m']
        features = expand_feats_stypes(features, canon4_stypes, norm_vars + cat_vars, single_vars)
        all_vars = list(features.columns)
        norm_vars = [x for x in all_vars if x.split(':')[0] in norm_vars]

        # train model
        model = models.BoundedLinearModel(len(all_vars) - 1)
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA_noredundant_6canon':
        # get data
        features = features[features['stype'].isin(canon6_stypes)]
        features = features.query('mir in @mirs5')
        nsites = utils.get_nsites(features)
        features['upper_bound'] = [upper_bound_dict[x] for x in features['stype']]
        features, cat_vars = one_hot_features(features, canon6_stypes)
        cat_vars = canon6_stypes

        # define vars
        single_vars = canon6_stypes + ['upper_bound']
        norm_vars = ['log_KA', 'TA', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
        features = expand_feats_stypes(features, canon6_stypes, norm_vars + cat_vars, single_vars)
        all_vars = list(features.columns)
        norm_vars = [x for x in all_vars if x.split(':')[0] in norm_vars]

        # train model
        model = models.BoundedLinearModel(len(all_vars) - 1)
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA_noredundant_6canon_sigmoid':
        # get data
        features = features[features['stype'].isin(canon6_stypes)]
        features = features.query('mir in @mirs5')
        nsites = utils.get_nsites(features)

        # define vars
        norm_vars = ['TA', 'Threep', 'SA', 'Local_AU', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
        all_vars = ['log_KA'] + norm_vars

        # train model
        model = models.SigmoidModel(2, len(all_vars) - 2, len(mirs5))
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid':
        # get data
        features = features[features['stype'].isin(canon6_stypes)]
        features = features.query('mir in @mirs5')
        nsites = utils.get_nsites(features)

        # define vars
        norm_vars = ['TA', 'Threep', 'SA', 'Local_AU', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
        all_vars = ['log_KA'] + norm_vars

        # train model
        model = models.DoubleSigmoidModel(2, len(all_vars) - 2, len(mirs5))
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_energy':
        # get data
        features = features[features['stype'].isin(canon6_stypes)]
        features = features.query('mir in @mirs5')
        nsites = utils.get_nsites(features)

        # define vars
        norm_vars = ['TA', 'Threep', 'SA', 'Local_AU', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
        all_vars = ['log_KA'] + norm_vars

        # train model
        model = models.DoubleSigmoidModel(3, len(all_vars) - 3, len(mirs5))
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_allsites':
        # get data
        features = features.query('mir in @mirs5')
        features = features[features['log_KA'] > 0]
        nsites = utils.get_nsites(features)

        # define vars
        norm_vars = ['TA', 'Threep', 'SA', 'Local_AU', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
        all_vars = ['log_KA'] + norm_vars

        # train model
        model = models.DoubleSigmoidModel(3, len(all_vars) - 3, len(mirs5))
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_allsites_noPCT':
        # get data
        features = features.query('mir in @mirs5')
        features = features[features['log_KA'] > 0]
        nsites = utils.get_nsites(features)

        # define vars
        norm_vars = ['TA', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'ORF_8m']
        all_vars = ['log_KA'] + norm_vars

        # train model
        model = models.DoubleSigmoidModel(2, len(all_vars) - 2, len(mirs5))
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_freeago_allsites':
        # get data
        features = features.query('mir in @mirs5')
        features = features[features['log_KA'] > 0]
        nsites = utils.get_nsites(features)

        # define vars
        norm_vars = ['TA', 'SA', 'Threep', 'Local_AU', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
        all_vars = ['log_KA'] + norm_vars

        # train model
        model = models.DoubleSigmoidFreeAGOModel(2, len(all_vars) - 2, len(mirs5))
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_freeago_allsites_noORF8m':
        # get data
        features = features.query('mir in @mirs5')
        features = features[features['log_KA'] > 0]
        nsites = utils.get_nsites(features)

        # define vars
        norm_vars = ['TA', 'Threep', 'SA', 'Local_AU', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT']
        all_vars = ['log_KA'] + norm_vars

        # train model
        model = models.DoubleSigmoidFreeAGOModel(2, len(all_vars) - 2, len(mirs5))
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_freeago_allsites_3PNB':
        # get data
        features = features.query('mir in @mirs5')
        features = features[features['log_KA'] > 0]
        nsites = utils.get_nsites(features)

        # define vars
        norm_vars = ['TA', 'Threep_NB', 'SA', 'Local_AU', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT', 'ORF_8m']
        all_vars = ['log_KA'] + norm_vars

        # train model
        model = models.DoubleSigmoidFreeAGOModel(2, len(all_vars) - 2, len(mirs5))
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'logKA_sigmoid_freeago_allsites':
        # get data
        features = features.query('mir in @mirs5')
        features = features[features['log_KA'] > 0]
        nsites = utils.get_nsites(features)

        # define vars
        norm_vars = []
        all_vars = ['log_KA'] + norm_vars

        # train model
        model = models.SigmoidFreeAGOModel(1, 0, len(mirs5))
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    elif options.MODEL_TYPE == 'logKA_UTRlen_doublesigmoid_freeago_allsites':
        # get data
        features = features.query('mir in @mirs5')
        features = features[features['log_KA'] > 0]
        nsites = utils.get_nsites(features)

        # define vars
        norm_vars = ['UTR_len']
        all_vars = ['log_KA'] + norm_vars

        # train model
        model = models.DoubleSigmoidFreeAGOModel(1, len(all_vars) - 1, len(mirs5))
        one_site, train_mirs, val_mirs = False, mirs5, mirs5

    else:
        raise ValueError('Invalid model type {}'.format(options.MODEL_TYPE))

    for ix in [1, 2, 3, 4, 5]:
        tf.reset_default_graph()
        model = models.DoubleSigmoidModel(ix, len(all_vars) - ix, len(mirs5))
        print('Num sigmoid: {}'.format(ix))

        train_r2s, val_r2s, pred_df = models.cross_val(tpms, features, nsites, train_mirs, val_mirs, all_vars,
                                                        norm_vars, model, 2000, one_site=one_site)
        print('Train r2 mean, std: {}, {}'.format(np.mean(train_r2s), np.std(train_r2s)))
        print('Val r2 mean, std:')
        print('{}, {}, {}'.format(options.MODEL_TYPE, np.mean(val_r2s), np.std(val_r2s)))
        print(all_vars)
        print(model.vars_evals)

    if options.OUT_FOLDER is not None:
        fig = plt.figure(figsize=(7,7))
        plt.scatter(pred_df['pred_normed'].values, pred_df['label_normed'].values, s=20)
        plt.savefig(os.path.join(options.OUT_FOLDER, options.MODEL_TYPE + '.png'))
        plt.close()

        pred_df.to_csv(os.path.join(options.OUT_FOLDER, options.MODEL_TYPE + '.tsv'), sep='\t', index=False)


