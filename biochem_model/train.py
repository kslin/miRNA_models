import copy
from optparse import OptionParser
import sys

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf

import utils
import models

pd.options.display.max_columns = 100


def split_vals(vals_4D, zero_indices):
    """
    Given a 4D matrix with ka values and features, split into ka_vals (3D), features (4D), and nosite_features (4D)
    """

    ka_vals_3D = vals_4D[:, :, :, 0]
    features_4D = vals_4D[:, :, :, 1:]
    nosite_features_4D = copy.copy(features_4D)
    for ix in zero_indices:
        nosite_features_4D[:, :, :, ix - 1] = 0

    return ka_vals_3D, features_4D, nosite_features_4D


def train_on_data(train_vals, test_vals_dict, test_guide, num_feats, mirna_data, passenger, outfile, set_vars={}):
    """
    Trains occupancy + context features model on data. Writes predictions to outfile
    """

    tf.reset_default_graph()

    # make placeholders for model
    ka_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='ka_vals')
    feature_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_feats], name='orf_ka')
    nosite_feature_tensor = tf.placeholder(tf.float32, shape=[None, None, None, NUM_FEATS], name='nosite_feats')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='mask')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')
    freeAGO_val = tf.placeholder(tf.float32, shape=[None, None, None], name='freeAGO_val')

    # make data dictionary
    train_data = {
        'ka_vals': ka_tensor,
        'mask': mask_tensor,
        'features': feature_tensor,
        'nosite_features': nosite_feature_tensor,
        'labels': labels_tensor,
        'passenger': passenger,
        'num_guides': len(train_vals['guides'])
    }

    # make feed dictionary
    train_feed_dict = {
        ka_tensor: train_vals['ka_vals_3D'],
        mask_tensor: train_vals['mask_3D'],
        feature_tensor: train_vals['features_4D'],
        nosite_feature_tensor: train_vals['nosite_features_4D'],
        labels_tensor: train_vals['labels']
    }

    # make and train model
    mod = models.OccupancyWithFeaturesModel(len(train_vals['guides']), num_feats, init_bound=True, passenger=passenger, set_vars=set_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mod.fit(sess, train_data, train_feed_dict, maxiter=200)
        print(mod.vars_evals)
        print(mod.r2)
        print(mod.final_loss)
        transcript_list = np.repeat(train_vals['transcripts'], len(train_vals['guides']))
        pred_df = pd.DataFrame({
            'transcript': transcript_list,
            'mir': list(train_vals['guides']) * len(train_vals['transcripts']),
            'pred': mod.eval_pred.flatten(),
            'label': mod.eval_label.flatten(),
            'pred_normed': mod.eval_pred_normed.flatten(),
            'label_normed': mod.eval_label_normed.flatten(),
        })

        # if outfile is given, write results to outfile
        if outfile is not None:
            pred_df.to_csv(outfile, sep='\t', index=False)

        # get freeAGO for test miRNA
        if passenger:
            current_freeAGO_all = mod.vars_evals['freeAGO'].reshape([-1, 2])
        else:
            current_freeAGO_all = mod.vars_evals['freeAGO'].reshape([-1, 1])

        print(current_freeAGO_all)

        if len(test_vals_dict) == 0:
            return

        # infer freeAGO of test miRNA from its target abundance
        train_guide_tas = mirna_data.loc[train_vals['guides']]['guide_TA'].values
        slope, inter = stats.linregress(train_guide_tas, current_freeAGO_all[:, 0])[:2]
        test_guide_ta = mirna_data.loc[test_guide]['guide_TA']
        new_freeago = [slope * test_guide_ta + inter]

        # infer freeAGO of test miRNA passenger strand from the median value from other miRNAs
        if passenger:
            new_freeago.append(np.median(current_freeAGO_all[:, 1]))

        current_freeAGO_all_val = np.concatenate([current_freeAGO_all, np.array([new_freeago])], axis=0).flatten().reshape([1, -1, 1])
        print(current_freeAGO_all)

        for key, test_vals in test_vals_dict.items():
            test_data = {
                'ka_vals': ka_tensor,
                'mask': mask_tensor,
                'features': feature_tensor,
                'nosite_features': nosite_feature_tensor,
                'labels': labels_tensor,
                'freeAGO': freeAGO_val,
                'passenger': passenger,
                'num_guides': len(test_vals['guides']),
            }

            test_feed_dict = {
                ka_tensor: test_vals['ka_vals_3D'],
                mask_tensor: test_vals['mask_3D'],
                feature_tensor: test_vals['features_4D'],
                nosite_feature_tensor: test_vals['nosite_features_4D'],
                freeAGO_val: current_freeAGO_all_val
            }

            test_preds = mod.predict(sess, test_data, test_feed_dict)
            test_labels = test_vals['labels']

            test_preds_normed = test_preds - np.mean(test_preds, axis=1).reshape([-1, 1])
            test_labels_normed = test_labels - np.mean(test_labels, axis=1).reshape([-1, 1])

            transcript_list = np.repeat(test_vals['transcripts'], len(test_vals['guides']))
            pred_df = pd.DataFrame({
                'transcript': transcript_list,
                'mir': list(test_vals['guides']) * len(test_vals['transcripts']),
                'pred': test_preds.flatten(),
                'label': test_labels.flatten(),
                'pred_normed': test_preds_normed.flatten(),
                'label_normed': test_labels_normed.flatten(),
            })

            # if outfile is given, write results to outfile
            if outfile is not None:
                pred_df.to_csv(outfile.replace('REPLACE', key), sep='\t', index=False)


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_file", dest="TPM_FILE", help="tpm data")
    parser.add_option("--feature_file", dest="FEATURE_FILE", help="file with features")
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--test_mir", dest="TEST_MIR", help="test miRNA", default=None)
    parser.add_option("--mode", dest="MODE", help="training_mode")
    parser.add_option("--extra_feats", dest="EXTRA_FEATS", help="comma-separated list of extra features", default=None)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--outfile", dest="OUTFILE", help="output file", default=None)

    (options, args) = parser.parse_args()

    if options.MODE not in ['all', 'canon', 'noncanon', 'shuffle_noncanon', 'only_shuffle_noncanon', 'train_only', 'test_only']:
        raise ValueError('Invalid mode.')

    # read miRNA DATA and get names of all guide miRNAs
    MIRNA_DATA = pd.read_csv(options.MIR_SEQS, sep='\t', index_col='mir')
    ALL_GUIDES = sorted(list(MIRNA_DATA.index))
    print(ALL_GUIDES)

    # split into training and testing
    if options.TEST_MIR is None:
        TRAIN_GUIDES = ALL_GUIDES
        TEST_GUIDES = ALL_GUIDES
    else:
        TRAIN_GUIDES = [x for x in ALL_GUIDES if x != options.TEST_MIR]
        TEST_GUIDES = TRAIN_GUIDES + [options.TEST_MIR]

    print(len(TRAIN_GUIDES), len(TEST_GUIDES))

    # if using passenger strand, add them
    if options.PASSENGER:
        TRAIN_MIRS = list(np.array([[x, x + '_pass'] for x in TRAIN_GUIDES]).flatten())
        TEST_MIRS = list(np.array([[x, x + '_pass'] for x in TEST_GUIDES]).flatten())
    else:
        TRAIN_MIRS = TRAIN_GUIDES
        TEST_MIRS = TEST_GUIDES

    # read in TPM data
    ALL_TPMS = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)
    ALL_TPMS.index.name = 'transcript'

    # subset into 10 batches
    NUM_BATCHES = 11
    ALL_TPMS['batch'] = [ix % NUM_BATCHES for ix in ALL_TPMS['ix']]

    # read in features
    ALL_FEATS = []
    for mir in TEST_MIRS:
        mir = mir.replace('*', '_pass')

        # read in features and apply filters
        temp = pd.read_csv(options.FEATURE_FILE.replace('MIR', mir), sep='\t')
        temp['mir'] = mir
        temp['log_KA'] = -1 * temp['log_kd']
        temp = temp[temp['log_KA'] > 0]  # filter for positive KAs
        ALL_FEATS.append(temp)

    ALL_FEATS = pd.concat(ALL_FEATS, sort=False)
    print(len(ALL_FEATS['mir'].unique()), len(ALL_FEATS['transcript'].unique()))

    print(ALL_FEATS['mir'].unique())
    print(TRAIN_MIRS, TEST_MIRS)

    # fill in SA_bg for noncanon sites
    if options.MODE == 'test_only':
        mean_SA_diff = 0.92177
    else:
        mean_SA_diff = np.nanmean(ALL_FEATS['logSA_diff'])
    print(f'Mean SA_diff: {mean_SA_diff}')
    ALL_FEATS['logSA_diff'] = ALL_FEATS['logSA_diff'].fillna(mean_SA_diff)
    ALL_FEATS['Threep_canon'] = ALL_FEATS['Threep'] * (ALL_FEATS['stype'] != 'no site')
    ALL_FEATS = ALL_FEATS.set_index(keys=['transcript', 'mir']).sort_index()

    if options.MODE in ['canon']:
        ALL_FEATS = ALL_FEATS[ALL_FEATS['stype'] != 'no site']  # only take canonical sites
    elif options.MODE in ['all', 'shuffle_noncanon', 'train_only', 'test_only']:
        ALL_FEATS = ALL_FEATS
        # ALL_FEATS = ALL_FEATS[(ALL_FEATS['stype'] != 'no site') | (~ALL_FEATS['in_ORF'])]  # get rid of noncanonical sites in ORF
    elif options.MODE in ['noncanon', 'only_shuffle_noncanon']:
        ALL_FEATS = ALL_FEATS[ALL_FEATS['stype'] == 'no site']
        # ALL_FEATS = ALL_FEATS[(ALL_FEATS['stype'] == 'no site') & (~ALL_FEATS['in_ORF'])] # only take noncanonical sites
    else:
        raise ValueError('invalid mode')

    NUM_SITES = ALL_FEATS.copy()
    NUM_SITES['nsites'] = 1
    NUM_SITES = NUM_SITES.groupby(['transcript', 'mir']).agg({'nsites': np.sum})

    # split transcripts into training and testing
    TRAIN_TRANSCRIPTS = list(ALL_TPMS.index)
    # TRAIN_TRANSCRIPTS = list(ALL_TPMS[ALL_TPMS['batch'] != 3].index)
    # TEST_TRANSCRIPTS = list(ALL_TPMS[ALL_TPMS['batch'] == 3].index)
    TEST_TRANSCRIPTS = list(ALL_TPMS.index)
    MAX_NSITES = np.max(NUM_SITES['nsites'])
    print(f'Max nsites: {MAX_NSITES}')
    print(len(TEST_TRANSCRIPTS))

    FEATURE_LIST = ['log_KA', 'in_ORF']
    if options.EXTRA_FEATS != 'none':
        FEATURE_LIST += options.EXTRA_FEATS.split(',')
    for feat in FEATURE_LIST:
        if feat not in ALL_FEATS.columns:
            raise ValueError(f'{feat} not a valid feature.')

    print(FEATURE_LIST)
    NUM_FEATS = len(FEATURE_LIST) - 1

    # ALL_FEATS = ALL_FEATS.query('transcript in @TRAIN_TRANSCRIPTS')

    print(np.sum(ALL_FEATS[FEATURE_LIST].values, axis=0))

    # get indices of features that do not affect background binding
    ZERO_INDICES = []
    for ix, feat in enumerate(FEATURE_LIST):
        if feat in ['Threep_canon', 'PCT']:
            ZERO_INDICES.append(ix)

    print(ZERO_INDICES)

    # expand features
    train_vals_4D, train_mask_3D = utils.expand_features_4D(TRAIN_TRANSCRIPTS, TRAIN_MIRS, MAX_NSITES,
                                                        FEATURE_LIST, ALL_FEATS)

    train_ka_vals_3D, train_features_4D, train_nosite_features_4D = split_vals(train_vals_4D, ZERO_INDICES)

    print(train_ka_vals_3D.shape, train_features_4D.shape, train_nosite_features_4D.shape, train_mask_3D.shape)
    print(np.sum(np.sum(train_mask_3D, axis=0), axis=1))
    print(np.sum(np.sum(np.sum(train_features_4D, axis=0), axis=0), axis=0))
    print(np.sum(np.sum(np.sum(train_nosite_features_4D, axis=0), axis=0), axis=0))

    print(np.sum(np.sum(train_features_4D, axis=0), axis=1).tolist())
    print(np.sum(np.sum(train_nosite_features_4D, axis=0), axis=1).tolist())


    train_vals = {
        'transcripts': TRAIN_TRANSCRIPTS,
        'guides': TRAIN_GUIDES,
        'ka_vals_3D': train_ka_vals_3D,
        'mask_3D': train_mask_3D,
        'features_4D': train_features_4D,
        'nosite_features_4D': train_nosite_features_4D,
        'labels': ALL_TPMS.loc[TRAIN_TRANSCRIPTS][TRAIN_GUIDES].values
    }

    if options.MODE == 'train_only':
        train_on_data(train_vals, {}, options.TEST_MIR, NUM_FEATS, MIRNA_DATA, options.PASSENGER, options.OUTFILE)

    if options.MODE == 'test_only':
        if options.PASSENGER:
            set_vars = {
                'log_decay': 0.5520938,
                'feature_coefs': np.array([[[[-1.7124393 ,  0.11626805,  0.12685744,  0.9295629 ]]]])
            }
        else:
            set_vars = {
                 'log_decay': 0.46782777,
                 'feature_coefs': np.array([[[[-1.6699566 ,  0.12071671,  0.07624636,  0.965269  ]]]])
            }

        train_on_data(train_vals, {}, options.TEST_MIR, NUM_FEATS, MIRNA_DATA, options.PASSENGER, options.OUTFILE, set_vars=set_vars)

    elif options.MODE in ['all', 'canon', 'noncanon']:  # this mode tests on a single test set
        test_vals_4D, test_mask_3D = utils.expand_features_4D(TEST_TRANSCRIPTS, TEST_MIRS, MAX_NSITES,
                                                        FEATURE_LIST, ALL_FEATS)

        test_ka_vals_3D, test_features_4D, test_nosite_features_4D = split_vals(test_vals_4D, ZERO_INDICES)

        test_vals = {
            'transcripts': TEST_TRANSCRIPTS,
            'guides': TEST_GUIDES,
            'ka_vals_3D': test_ka_vals_3D,
            'mask_3D': test_mask_3D,
            'features_4D': test_features_4D,
            'nosite_features_4D': test_nosite_features_4D,
            'labels': ALL_TPMS.loc[TEST_TRANSCRIPTS][TEST_GUIDES].values
        }

        train_on_data(train_vals, {'0': test_vals}, options.TEST_MIR, NUM_FEATS, MIRNA_DATA, options.PASSENGER, options.OUTFILE)

    elif options.MODE == 'shuffle_noncanon':  # this mode tests on many shuffled test sets
        canon_feats = ALL_FEATS[ALL_FEATS['stype'] != 'no site']
        NUM_CANON_SITES = canon_feats.copy()
        NUM_CANON_SITES['nsites'] = 1
        NUM_CANON_SITES = NUM_CANON_SITES.groupby(['transcript', 'mir']).agg({'nsites': np.sum})
        max_canon_nsites = np.max(NUM_CANON_SITES['nsites'])

        noncanon_feats = ALL_FEATS[ALL_FEATS['stype'] == 'no site']
        NUM_NONCANON_SITES = noncanon_feats.copy()
        NUM_NONCANON_SITES['nsites'] = 1
        NUM_NONCANON_SITES = NUM_NONCANON_SITES.groupby(['transcript', 'mir']).agg({'nsites': np.sum})
        max_noncanon_nsites = np.max(NUM_NONCANON_SITES['nsites'])

        canon_test_vals_4D, canon_test_mask_3D = utils.expand_features_4D(TEST_TRANSCRIPTS, TEST_MIRS, max_canon_nsites,
                                                        FEATURE_LIST, canon_feats)

        canon_test_ka_vals_3D, canon_test_features_4D, canon_test_nosite_features_4D = split_vals(canon_test_vals_4D, ZERO_INDICES)

        noncanon_test_vals_4D, noncanon_test_mask_3D = utils.expand_features_4D(TEST_TRANSCRIPTS, TEST_MIRS, max_noncanon_nsites,
                                                        FEATURE_LIST, noncanon_feats)

        test_vals_dict = {}
        for ix, shuffle_guide in enumerate(TRAIN_GUIDES):
            if options.PASSENGER:
                new_order = list(np.arange(len(TRAIN_GUIDES) * 2)) + [ix * 2, (ix * 2) + 1]
            else:
                new_order = list(np.arange(len(TRAIN_GUIDES))) + [ix]

            noncanon_test_vals_4D_shuffled = copy.copy(noncanon_test_vals_4D)[:, new_order, :, :]
            noncanon_test_ka_vals_3D_shuffled, noncanon_test_features_4D_shuffled, noncanon_test_nosite_features_4D_shuffled = split_vals(noncanon_test_vals_4D_shuffled, ZERO_INDICES)
            noncanon_test_mask_3D_shuffled = copy.copy(noncanon_test_mask_3D[:, new_order, :])

            test_ka_vals_3D = np.concatenate([canon_test_ka_vals_3D, noncanon_test_ka_vals_3D_shuffled], axis=2)
            test_features_4D = np.concatenate([canon_test_features_4D, noncanon_test_features_4D_shuffled], axis=2)
            test_nosite_features_4D = np.concatenate([canon_test_nosite_features_4D, noncanon_test_nosite_features_4D_shuffled], axis=2)
            test_mask_3D = np.concatenate([canon_test_mask_3D, noncanon_test_mask_3D_shuffled], axis=2)

            test_vals_dict[shuffle_guide] = {
                'transcripts': TEST_TRANSCRIPTS,
                'guides': TEST_GUIDES,
                'ka_vals_3D': test_ka_vals_3D,
                'mask_3D': test_mask_3D,
                'features_4D': test_features_4D,
                'nosite_features_4D': test_nosite_features_4D,
                'labels': ALL_TPMS.loc[TEST_TRANSCRIPTS][TEST_GUIDES].values
            }

            train_on_data(train_vals, test_vals_dict, options.TEST_MIR, NUM_FEATS, MIRNA_DATA, options.PASSENGER, options.OUTFILE)

    else:  # this mode tests on many shuffled test sets
        test_vals_4D, test_mask_3D = utils.expand_features_4D(TEST_TRANSCRIPTS, TEST_MIRS, MAX_NSITES,
                                                        FEATURE_LIST, ALL_FEATS)

        test_vals_dict = {}
        for ix, shuffle_guide in enumerate(TRAIN_GUIDES):
            if options.PASSENGER:
                new_order = list(np.arange(len(TRAIN_GUIDES) * 2)) + [ix * 2, (ix * 2) + 1]
            else:
                new_order = list(np.arange(len(TRAIN_GUIDES))) + [ix]

            test_vals_4D_shuffled = copy.copy(test_vals_4D)[:, new_order, :, :]
            test_ka_vals_3D_shuffled, test_features_4D_shuffled, test_nosite_features_4D_shuffled = split_vals(test_vals_4D_shuffled, ZERO_INDICES)
            test_mask_3D_shuffled = copy.copy(test_mask_3D[:, new_order, :])

            test_vals_dict[shuffle_guide] = {
                'transcripts': TEST_TRANSCRIPTS,
                'guides': TEST_GUIDES,
                'ka_vals_3D': test_ka_vals_3D_shuffled,
                'mask_3D': test_mask_3D_shuffled,
                'features_4D': test_features_4D_shuffled,
                'nosite_features_4D': test_nosite_features_4D_shuffled,
                'labels': ALL_TPMS.loc[TEST_TRANSCRIPTS][TEST_GUIDES].values
            }

            train_on_data(train_vals, test_vals_dict, options.TEST_MIR, NUM_FEATS, MIRNA_DATA, options.PASSENGER, options.OUTFILE)
