from optparse import OptionParser
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf

import utils
import models


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_file", dest="TPM_FILE", help="tpm data")
    parser.add_option("--feature_dir", dest="FEATURE_DIR", help="feature data")
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--kd_dir", dest="KD_DIR", help="KD predictions")
    parser.add_option("--test_mir", dest="TEST_MIR", help="test miRNA")
    parser.add_option("--extra_feats", dest="EXTRA_FEATS", help="comma-separated list of extra features")
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--outfile", dest="OUTFILE", help="output file", default=None)

    (options, args) = parser.parse_args()

    all_guides = sorted(['mir1', 'mir124', 'mir155', 'mir7', 'lsy6', 'mir153', 'mir139', 'mir144', 'mir223', 'mir137',
                    'mir205', 'mir143', 'mir182', 'mir199a', 'mir204', 'mir216b'])
    train_guides = [x for x in all_guides if x != options.TEST_MIR]
    test_guides = train_guides + [options.TEST_MIR]

    print(len(train_guides), len(test_guides))

    ### READ miRNA DATA ###
    MIRNA_DATA = pd.read_csv(options.MIR_SEQS, sep='\t', index_col='mir')

    if options.PASSENGER:
        train_mirs = list(np.array([[x, x + '_pass'] for x in train_guides]).flatten())
        test_mirs = list(np.array([[x, x + '_pass'] for x in test_guides]).flatten())
    else:
        train_mirs = train_guides
        test_mirs = test_guides

    all_tpms = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)
    all_tpms.index.name = 'transcript'

    # shuffle transcripts and subset into 10 batches
    num_batches = 11
    all_tpms['batch'] = [ix % num_batches for ix in all_tpms['ix']]

    all_transcripts = list(all_tpms.index)
    batch_ixs = {b: all_tpms[all_tpms['batch'] == b]['ix'].values for b in range(num_batches)}

    all_features = []
    for mir in test_mirs:
        temp = pd.read_csv(os.path.join(options.FEATURE_DIR, f'{mir}.txt'), sep='\t')
        temp['mir'] = mir
        kds_temp = pd.read_csv(os.path.join(options.KD_DIR, f'{mir}_kds.txt'), sep='\t', index_col='12mer')
        temp['log_KA'] = -1 * kds_temp.reindex(temp['12mer'].values)['log_kd'].values

        if len(temp) != len(temp.dropna(subset=['log_KA'])):
            raise ValueError(f"not all 12mers match for {mir}")
        temp = temp[temp['log_KA'] > 0]
        temp = temp[(temp['stype'] != 'no site') | (~temp['in_ORF'])]
        # temp = temp[temp['stype'] != 'no site']
        all_features.append(temp)

    all_features = pd.concat(all_features, sort=False)
    all_features['Threep_canon'] = (all_features['stype'].values != 'no site') * all_features['Threep']

    mean_SA_diff = np.nanmean(all_features['logSA_diff'])
    print(f'Mean SA_diff: {mean_SA_diff}')
    all_features['logSA_diff'] = all_features['logSA_diff'].fillna(mean_SA_diff)
    all_features = all_features.reset_index().set_index(keys=['transcript', 'mir']).sort_index()

    NUM_SITES = all_features.copy()
    NUM_SITES['nsites'] = 1
    NUM_SITES = NUM_SITES.groupby(['transcript', 'mir']).agg({'nsites': np.sum})

    train_transcripts = list(all_tpms[all_tpms['batch'] != 3].index)
    test_transcripts = list(all_tpms[all_tpms['batch'] == 3].index)
    max_nsites = np.max(NUM_SITES['nsites'])
    print(f'Max nsites: {max_nsites}')

    print(all_features.head())

    FEATURE_LIST = ['log_KA', 'in_ORF']
    FEATURE_LIST += options.EXTRA_FEATS.split(',')
    for feat in FEATURE_LIST:
        if feat not in all_features.columns:
            raise ValueError(f'{feat} not a valid feature.')

    print(FEATURE_LIST)
    train_features_4D, train_mask_3D = utils.expand_features_4D(train_transcripts, train_mirs, max_nsites,
                                                        FEATURE_LIST, all_features)

    test_features_4D, test_mask_3D = utils.expand_features_4D(test_transcripts, test_mirs, max_nsites,
                                                        FEATURE_LIST, all_features)

    print(train_features_4D.shape, train_mask_3D.shape)
    print(test_features_4D.shape, test_mask_3D.shape)
    print(np.sum(np.sum(train_mask_3D, axis=0), axis=1))
    print(np.sum(np.sum(test_mask_3D, axis=0), axis=1))

    NUM_FEATS = len(FEATURE_LIST) - 1

    ka_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='ka_vals')
    feature_tensor = tf.placeholder(tf.float32, shape=[None, None, None, NUM_FEATS], name='orf_ka')
    mask_tensor = tf.placeholder(tf.float32, shape=[None, None, None], name='mask')
    labels_tensor = tf.placeholder(tf.float32, shape=[None, None], name='labels')

    freeAGO_val = tf.placeholder(tf.float32, shape=[None, None, None], name='freeAGO_val')

    train_data = {
        'ka_vals': ka_tensor,
        'mask': mask_tensor,
        'features': feature_tensor,
        'labels': labels_tensor,
        'passenger': options.PASSENGER,
        'num_guides': len(train_guides)
    }

    train_feed_dict = {
        ka_tensor: train_features_4D[:, :, :, 0],
        mask_tensor: train_mask_3D,
        feature_tensor: train_features_4D[:, :, :, 1:],
        labels_tensor: all_tpms.loc[train_transcripts][train_guides].values
    }

    mod = models.OccupancyWithFeaturesModel(len(train_mirs), NUM_FEATS, init_bound=True, passenger=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mod.fit(sess, train_data, train_feed_dict, maxiter=200)
        print(mod.vars_evals)

        # get freeAGO for test miRNA
        if options.PASSENGER:
            current_freeAGO_all = mod.vars_evals['freeAGO'].reshape([-1, 2])
        else:
            current_freeAGO_all = mod.vars_evals['freeAGO'].reshape([-1, 1])

        # infer freeAGO of test miRNA from its target abundance
        train_guide_tas = MIRNA_DATA.loc[train_guides]['guide_TA'].values
        slope, inter = stats.linregress(train_guide_tas, current_freeAGO_all[:, 0])[:2]
        test_guide_ta = MIRNA_DATA.loc[options.TEST_MIR]['guide_TA']
        new_freeago = [slope * test_guide_ta + inter]

        # infer freeAGO of test miRNA passenger strand from the median value from other miRNAs
        if options.PASSENGER:
            new_freeago.append(np.median(current_freeAGO_all[:, 1]))

        current_freeAGO_all_val = np.concatenate([current_freeAGO_all, np.array([new_freeago])], axis=0).flatten().reshape([1, -1, 1])
        print(current_freeAGO_all)

        test_data = {
            'ka_vals': ka_tensor,
            'mask': mask_tensor,
            'features': feature_tensor,
            'labels': labels_tensor,
            'freeAGO': freeAGO_val,
            'passenger': options.PASSENGER,
            'num_guides': len(test_guides),
        }

        test_feed_dict = {
            ka_tensor: test_features_4D[:, :, :, 0],
            mask_tensor: test_mask_3D,
            feature_tensor: test_features_4D[:, :, :, 1:],
            freeAGO_val: current_freeAGO_all_val
        }

        test_preds = mod.predict(sess, test_data, test_feed_dict)
        test_labels = all_tpms.loc[test_transcripts][test_guides].values

        test_preds_normed = test_preds - np.mean(test_preds, axis=1).reshape([-1, 1])
        test_labels_normed = test_labels - np.mean(test_labels, axis=1).reshape([-1, 1])

        transcript_list = np.repeat(test_transcripts, len(test_guides))
        pred_df = pd.DataFrame({
            'transcript': transcript_list,
            'mir': list(test_guides) * len(test_transcripts),
            'pred': test_preds.flatten(),
            'label': test_labels.flatten(),
            'pred_normed': test_preds_normed.flatten(),
            'label_normed': test_labels_normed.flatten(),
        })

        if options.OUTFILE is not None:
            pred_df.to_csv(options.OUTFILE, sep='\t', index=False)

        temp = pred_df[pred_df['mir'] == options.TEST_MIR]
        print(stats.linregress(temp['pred_normed'], temp['label_normed']))
        print(stats.linregress(temp['pred_normed'], temp['label_normed'])[2]**2)
