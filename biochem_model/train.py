import json
from optparse import OptionParser
import sys

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf

import predict
import utils
import models

pd.options.display.max_columns = 100


def train_on_data(train_vals, num_feats, passenger, outfile, init_bound, set_vars={}):
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
    utr3_length_tensor = tf.placeholder(tf.float32, shape=[None, 1], name='utr3_length')
    orf_length_tensor = tf.placeholder(tf.float32, shape=[None, 1], name='orf_length')

    # make data dictionary
    train_data = {
        'ka_vals': ka_tensor,
        'mask': mask_tensor,
        'features': feature_tensor,
        'nosite_features': nosite_feature_tensor,
        'labels': labels_tensor,
        'passenger': passenger,
        'num_guides': len(train_vals['guides']),
        'utr3_length': utr3_length_tensor,
        'orf_length': orf_length_tensor, 
        # 'freeAGO': -5.95 # uncomment for using TA to fit
    }

    # make feed dictionary
    train_feed_dict = {
        ka_tensor: train_vals['ka_vals_3D'],
        mask_tensor: train_vals['mask_3D'],
        feature_tensor: train_vals['features_4D'],
        nosite_feature_tensor: train_vals['nosite_features_4D'],
        labels_tensor: train_vals['labels'],
        utr3_length_tensor: train_vals['utr3_length'],
        orf_length_tensor: train_vals['orf_length']
    }

    # set_vars.update({'set_freeAGO': -5.52})

    # make and train model
    mod = models.OccupancyWithFeaturesModel(len(train_vals['guides']), num_feats, init_bound=init_bound, fit_background=False, passenger=passenger, set_vars=set_vars)
    # mod = models.OccupancyWithFeaturesWithLensModel(len(train_vals['guides']), num_feats, passenger=passenger, set_vars=set_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mod.fit(sess, train_data, train_feed_dict, maxiter=200)
        print(f'Train r2: {mod.r2}')
        print(f'Train loss: {mod.final_loss}')
        print(f'Fit params: {mod.vars_evals}')
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

        temp = pred_df[pred_df['mir'] == 'mir190a']
        if len(temp) > 0:
            print(stats.linregress(temp['pred_normed'], temp['label_normed'])[2]**2)

        mod.vars_evals['r2'] = mod.r2
        mod.vars_evals['final_loss'] = mod.final_loss

    return mod.vars_evals


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_file", dest="TPM_FILE", help="tpm data")
    parser.add_option("--feature_file", dest="FEATURE_FILE", help="file with features")
    parser.add_option("--mir_to_shuffle", dest="MIR_TO_SHUFFLE", help="miRNA to shuffle", default=None)
    parser.add_option("--shuffle_mir", dest="SHUFFLE_MIR", help="miRNA to shuffle in", default=None)
    parser.add_option("--shuffle_file", dest="SHUFFLE_FILE", help="file with shuffled features", default=None)
    parser.add_option("--kd_cutoff", dest="KD_CUTOFF", help="cutoff value for KDs", default=None, type=float)
    parser.add_option("--setparams", dest="SETPARAMS", help="json file of parameters to set", default=None)
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--test_mir", dest="TEST_MIR", help="test miRNA", default=None)
    parser.add_option("--mode", dest="MODE", help="training_mode")
    parser.add_option("--init_bound", dest="INIT_BOUND", help="offset by background binding", default=False, action='store_true')
    parser.add_option("--extra_feats", dest="EXTRA_FEATS", help="comma-separated list of extra features", default=None)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--set_freeago", dest="SET_FREEAGO", help="infer freeAGO", default=False, action='store_true')
    parser.add_option("--outfile", dest="OUTFILE", help="output file", default=None)
    parser.add_option("--outparams", dest="OUTPARAMS", help="output file for writing fitted parameters")

    (options, args) = parser.parse_args()

    if options.SHUFFLE_MIR is not None:
        if options.MIR_TO_SHUFFLE == options.SHUFFLE_MIR:
            print(options.MIR_TO_SHUFFLE, options.SHUFFLE_MIR)
            sys.exit()

    if options.MODE not in ['all', 'canon', 'noncanon']:
        raise ValueError('Invalid mode.')

    if options.EXTRA_FEATS == 'none':
        MODEL = 'biochem'
    elif options.EXTRA_FEATS == 'logSA_diff,Threep_canon,PCT':
        MODEL = 'biochemplus'
    else:
        MODEL = options.EXTRA_FEATS.replace(',','_')

    # read miRNA DATA and get names of all guide miRNAs
    MIRNA_DATA = pd.read_csv(options.MIR_SEQS, sep='\t', index_col='mir')
    ALL_GUIDES = sorted(list(MIRNA_DATA.index))
    # print(ALL_GUIDES)

    # split into training and testing
    if options.TEST_MIR is None:
        TRAIN_GUIDES = ALL_GUIDES
    else:
        TRAIN_GUIDES = [x for x in ALL_GUIDES if x != options.TEST_MIR]

    print(f'Number training guides: {len(TRAIN_GUIDES)}')

    # if using passenger strand, add them
    if options.PASSENGER:
        TRAIN_MIRS = list(np.array([[x, x + '_pass'] for x in TRAIN_GUIDES]).flatten())
    else:
        TRAIN_MIRS = TRAIN_GUIDES

    # read in TPM data
    ALL_TPMS = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)
    ALL_TPMS.index.name = 'transcript'
    TRAIN_TRANSCRIPTS = list(ALL_TPMS.index)

    # read in features for all miRNAs
    ALL_FEATS = []
    for mir in TRAIN_MIRS:
        mir = mir.replace('*', '_pass')  # sanitize miRNA name
        if mir.replace('_pass', '') == options.MIR_TO_SHUFFLE:
            if '_pass' in mir:
                shuffle = options.SHUFFLE_MIR + '_pass'
            else:
                shuffle = options.SHUFFLE_MIR
            print(options.SHUFFLE_FILE.replace('MIR', mir).replace('SHUFFLE', shuffle))
            temp = pd.read_csv(options.SHUFFLE_FILE.replace('MIR', mir).replace('SHUFFLE', shuffle), sep='\t')
        else:
            temp = pd.read_csv(options.FEATURE_FILE.replace('MIR', mir), sep='\t')
        temp['mir'] = mir

        # fill in SA_bg for noncanon sites
        mean_SA_diff = np.nanmean(temp['logSA_diff'])
        temp['logSA_diff'] = temp['logSA_diff'].fillna(mean_SA_diff)
        ALL_FEATS.append(temp)

    ALL_FEATS = pd.concat(ALL_FEATS, sort=False)

    # convert KD to KA
    ALL_FEATS['log_KA'] = -1 * ALL_FEATS['log_kd']

    # only use 3p-pairing score for canonical sites
    ALL_FEATS['Threep_canon'] = ALL_FEATS['Threep'] * (ALL_FEATS['stype'] != 'no site')

    # apply KD cutoff if given
    if options.KD_CUTOFF is not None:
        ALL_FEATS = ALL_FEATS[ALL_FEATS['log_kd'] < options.KD_CUTOFF]

    print('Total number of transcripts: {}'.format(len(ALL_FEATS['transcript'].unique())))
    print('Total number of miRNAs: {}'.format(len(ALL_FEATS['mir'].unique())))
    print(ALL_FEATS['mir'].unique())
    ALL_FEATS = ALL_FEATS.set_index(keys=['transcript', 'mir']).sort_index()

    if options.MODE in ['canon']:
        ALL_FEATS = ALL_FEATS[ALL_FEATS['stype'] != 'no site']  # only take canonical sites
    elif options.MODE in ['all']:
        ALL_FEATS = ALL_FEATS
    elif options.MODE in ['noncanon']:
        ALL_FEATS = ALL_FEATS[ALL_FEATS['stype'] == 'no site']
    else:
        raise ValueError('invalid mode')

    ALL_FEATS['intercept'] = 1.0

    NUM_SITES = ALL_FEATS.copy()
    NUM_SITES['nsites'] = 1
    NUM_SITES = NUM_SITES.groupby(['transcript', 'mir']).agg({'nsites': np.sum})
    MAX_NSITES = np.max(NUM_SITES['nsites'])
    print(f'Max nsites: {MAX_NSITES}')

    FEATURE_LIST = ['log_KA', 'in_ORF']
    if options.EXTRA_FEATS != 'none':
        FEATURE_LIST += options.EXTRA_FEATS.split(',')
    for feat in FEATURE_LIST:
        if feat not in ALL_FEATS.columns:
            raise ValueError(f'{feat} not a valid feature.')

    print(FEATURE_LIST)
    NUM_FEATS = len(FEATURE_LIST) - 1

    print(np.sum(ALL_FEATS[FEATURE_LIST].values, axis=0))

    if options.SET_FREEAGO:
        FITTED_PARAMS, MEAN_FA_GUIDE, MEAN_FA_PASS = predict.get_params(options.SETPARAMS.replace('MODEL', MODEL), options.PASSENGER)
        pred_df = []
        for mir in TRAIN_GUIDES:
            seed = MIRNA_DATA.loc[mir]['guide_seq'][1:7]
            temp1 = ALL_FEATS.query('mir == @mir')
            temp1['freeAGO'] = MEAN_FA_GUIDE
            if 'CG' in seed:  # adjust for miRNAs with CpGs
                temp1['freeAGO'] += 1

            if options.PASSENGER:
                seed = MIRNA_DATA.loc[mir]['pass_seq'][1:7]
                pass_mir = mir + '_pass'
                temp2 = ALL_FEATS.query('mir == @pass_mir')
                temp2['freeAGO'] = MEAN_FA_PASS
                temp = pd.concat([temp1, temp2], sort=False)
            else:
                temp = temp1
                
            if options.EXTRA_FEATS == 'none':
                FEATURE_LIST = ''
            else:
                FEATURE_LIST = options.EXTRA_FEATS

            # predict using fitted parameters
            _, temp_pred = predict.predict(temp, FEATURE_LIST, FITTED_PARAMS)

            temp_pred = temp_pred[['pred']].reindex(TRAIN_TRANSCRIPTS).fillna(0)
            temp_pred.columns = [mir]

            pred_df.append(temp_pred)

        pred_df = pd.concat(pred_df, axis=1, join='inner', sort=False)
        pred_df.index.name = 'transcript'
        pred_df['mean_pred'] = np.mean(pred_df.values, axis=1)

        pred_df_long = pred_df.reset_index().melt(id_vars=['transcript'], value_vars=TRAIN_GUIDES)
        pred_df_long.columns = ['transcript','mir','pred']
        pred_df_long['pred_normed'] = pred_df_long['pred'] - pred_df.reindex(pred_df_long['transcript'].values)['mean_pred'].values
        pred_df_long.to_csv(options.OUTFILE.replace('MODEL', MODEL), sep='\t', index=False)

    else:

        # get indices of features that do not affect background binding
        ZERO_INDICES = []
        for ix, feat in enumerate(FEATURE_LIST):
            if feat in ['Threep_canon', 'PCT']:
                ZERO_INDICES.append(ix)

        print(ZERO_INDICES)

        # if indicated, read in parameters to set
        if options.SETPARAMS is not None:
            with open(options.SETPARAMS.replace('MODEL', MODEL), 'r') as infile:
                setparams = json.load(infile)
                setparams['feature_coefs'] = np.array(setparams['feature_coefs']).reshape([1, 1, 1, -1])
                print(setparams)
        else:
            setparams = {}

        # expand features
        train_vals_4D, train_mask_3D = utils.expand_features_4D(TRAIN_TRANSCRIPTS, TRAIN_MIRS, MAX_NSITES,
                                                            FEATURE_LIST, ALL_FEATS)

        train_ka_vals_3D, train_features_4D, train_nosite_features_4D = utils.split_vals(train_vals_4D, ZERO_INDICES)

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
            'labels': ALL_TPMS.loc[TRAIN_TRANSCRIPTS][TRAIN_GUIDES].values,
            'utr3_length': ALL_TPMS.loc[TRAIN_TRANSCRIPTS][['utr3_length']].values / 5000,
            'orf_length': ALL_TPMS.loc[TRAIN_TRANSCRIPTS][['orf_length']].values / 5000,
        }


        params = train_on_data(train_vals, NUM_FEATS, options.PASSENGER, options.OUTFILE.replace('MODEL', MODEL), options.INIT_BOUND, setparams)
        params['freeAGO'] = params['freeAGO'].flatten().tolist()
        params['feature_coefs'] = params['feature_coefs'].flatten().tolist()
        params['FEATURE_LIST'] = FEATURE_LIST
        params['PASSENGER'] = options.PASSENGER
        params['KD_CUTOFF'] = options.KD_CUTOFF
        params['TRAIN_MIRS'] = TRAIN_MIRS

        # convert all numpy types to native python types
        for key, val in params.items():
            try:
                params[key] = val.item()
            except AttributeError:
                continue

        with open(options.OUTPARAMS.replace('MODEL', MODEL), 'w') as outparams:
            json.dump(params, outparams)

