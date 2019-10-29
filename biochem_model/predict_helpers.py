import bisect
import json

import numpy as np
import pandas as pd

import utils


def get_params(param_file, passenger):
    # read in parameters
    with open(param_file, 'r') as infile:
        TRAIN_PARAMS = json.load(infile)

    FITTED_PARAMS = {'log_decay': TRAIN_PARAMS['log_decay']}
    for feat, val in zip(TRAIN_PARAMS['FEATURE_LIST'][1:], TRAIN_PARAMS['feature_coefs']):
        FITTED_PARAMS[feat + '_coef'] = val

    for param in ['nosite_conc', 'utr3_coef', 'orf_coef']:
        if param in TRAIN_PARAMS:
            FITTED_PARAMS[param] = TRAIN_PARAMS[param]
    FITTED_PARAMS

    guide_freeagos = [y for (x,y) in zip(TRAIN_PARAMS['TRAIN_MIRS'], TRAIN_PARAMS['freeAGO']) if ('_pass' not in x)]
    MEAN_FA_GUIDE = np.mean(guide_freeagos)
    
    if passenger:
        pass_freeagos = [y for (x,y) in zip(TRAIN_PARAMS['TRAIN_MIRS'], TRAIN_PARAMS['freeAGO']) if ('_pass' in x)]
        MEAN_FA_PASS = np.mean(pass_freeagos)
    else:
        MEAN_FA_PASS = None

    return FITTED_PARAMS, MEAN_FA_GUIDE, MEAN_FA_PASS


def process_features(filename, kd_cutoff=None, transcripts=None):

    # read in data
    features = pd.read_csv(filename, sep='\t')
    if transcripts is not None:
        features = features[features['transcript'].isin(transcripts)]

    mean_logSA_diff = np.nanmean(features['logSA_diff'].values)
    features['logSA_diff'] = features['logSA_diff'].fillna(mean_logSA_diff)

    features['Threep_canon'] = (features['stype'] != 'no site') * (features['Threep'])
    features['log_KA'] = -1 * features['log_kd']

    if kd_cutoff is not None:
        features = features[features['log_kd'] < kd_cutoff]

    return features


def predict(features, feature_list, fitted_params):
    """Predict logFC values from feature dataframe"""

    features_copy = features.copy()
    feature_list = feature_list.split(',')
    features_copy['offset'] = 0
    features_copy['offset_all'] = fitted_params['in_ORF_coef'] * features_copy['in_ORF']

    if 'logSA_diff' in feature_list:
        features_copy['offset_all'] += fitted_params['logSA_diff_coef'] * features_copy['logSA_diff']
    if 'Threep_canon' in feature_list:
        features_copy['offset'] += fitted_params['Threep_canon_coef'] * features_copy['Threep_canon']
    if 'PCT' in feature_list:
        features_copy['offset'] += fitted_params['PCT_coef'] * features_copy['PCT']
    
    features_copy['occ'] = utils.sigmoid(features_copy['freeAGO'] + features_copy['log_KA'] + features_copy['offset'] + features_copy['offset_all'])
    features_copy['occ_init'] = utils.sigmoid(features_copy['freeAGO'] + fitted_params['nosite_conc'] + features_copy['offset_all'])

    decay = np.exp(fitted_params['log_decay'])

    pred = features_copy.groupby(['transcript']).agg({'occ': np.sum, 'occ_init': np.sum})
    pred['pred'] = np.log1p(decay * pred['occ_init']) - np.log1p(decay * pred['occ'])
    return features_copy, pred


def predict_transcript_lengths(features, lengths, feature_list, fitted_params):
    """Predict logFC values from feature dataframe"""

    features_copy = features.copy()
    feature_list = feature_list.split(',')
    features_copy['offset'] = fitted_params['in_ORF_coef'] * features_copy['in_ORF']

    if 'logSA_diff' in feature_list:
        features_copy['offset'] += fitted_params['logSA_diff_coef'] * features_copy['logSA_diff']
    if 'Threep_canon' in feature_list:
        features_copy['offset'] += fitted_params['Threep_canon_coef'] * features_copy['Threep_canon']
    if 'PCT' in feature_list:
        features_copy['offset'] += fitted_params['PCT_coef'] * features_copy['PCT']
    if 'TA' in feature_list:
        features_copy['offset'] += fitted_params['TA_coef'] * features_copy['TA']
    if 'passenger' in feature_list:
        features_copy['offset'] += fitted_params['passenger_coef'] * features_copy['passenger']
    if 'intercept' in feature_list:
        features_copy['offset'] += fitted_params['intercept_coef'] * features_copy['intercept']
    
    features_copy['occ'] = utils.sigmoid(features_copy['freeAGO'] + features_copy['log_KA'] + features_copy['offset'])

    decay = np.exp(fitted_params['log_decay'])

    pred = features_copy.groupby(['transcript']).agg({'occ': np.sum})
    pred = pd.concat([pred, lengths], axis=1, join_axes=[lengths.index]).fillna(0)
    pred['occ_with_length'] = pred['occ'] + (np.exp(fitted_params['utr3_coef']) * pred['utr3_length']) + (np.exp(fitted_params['orf_coef']) * pred['orf_length'])
    pred['pred'] = -1 * np.log1p(decay * pred['occ_with_length'])
    return features_copy, pred


def add_bin_info(features, AIR_dict):
    temp = features.copy()
    temp['utr_loc'] = temp['loc'] - temp['orf_length'] + 6
    groups = []
    for transcript, group in temp.groupby('transcript'):
        AIR_temp = AIR_dict[transcript]
        ixs = [bisect.bisect_left(AIR_temp['stop'].values, x) for x in group['utr_loc'].values]
        group['bin'] = ixs
        groups.append(group)
    features_with_bin = pd.concat(groups)
    return features_with_bin


def predict_withAIR(features_with_bin, feature_list, fitted_params, AIRS):
    """Predict logFC values from feature dataframe with alternative isoform ratio information"""

    decay = np.exp(fitted_params['log_decay'])

    temp = features_with_bin.copy()
    temp, _ = predict(temp, feature_list, fitted_params)
    temp = temp[['transcript','bin','occ','occ_init']].groupby(['transcript', 'bin']).agg(np.sum)
    unique_transcripts = features_with_bin['transcript'].unique()

    temp2 = AIRS[AIRS['transcript'].isin(unique_transcripts)].set_index(['transcript', 'bin']).sort_index()
    temp2 = pd.concat([temp, temp2], axis=1, join_axes=[temp2.index]).reset_index().fillna(0)

    results = []
    for transcript, group in temp2.groupby('transcript'):
        group['occ'] = np.cumsum(group['occ'])
        group['occ_init'] = np.cumsum(group['occ_init'])
        group['pred'] = np.log1p(decay*group['occ_init']) - np.log1p(decay*group['occ'])
        if np.abs(np.sum(group['percent_end_here']) - 1) > 0.0001:
            print(np.sum(group['percent_end_here']))
            raise ValueError()
        results.append([transcript, np.sum(group['pred'] * group['percent_end_here'])])

    results = pd.DataFrame(results, columns=['transcript', f'pred']).set_index('transcript')
    return temp2, results


def process_TS7_features(filename, mirs, transcripts):
    upper_bound_dict = {
        '8mer-1a': -0.03,
        '7mer-m8': -0.02,
        '7mer-1a': -0.01,
        '6mer': 0.0
    }

    features = pd.read_csv(filename, sep='\t')
    features = features.rename(columns={'miRNA family': 'mir'})
    features.columns = [x.replace(' ','_') for x in features.columns]
    features = features.rename(columns={'Gene_ID': 'transcript'})
    features = features[features['mir'].isin(mirs)]
    features = features[features['transcript'].isin(transcripts)]
    features = features.set_index(keys=['transcript','mir']).sort_index()
    ts7_stypes = list(features['Site_type'].unique())
    for stype in ts7_stypes:
        features[stype] = (features['Site_type'] == stype).astype(float)

    features['upper_bound'] = [upper_bound_dict[x] for x in features['Site_type']]
    return features


def calculate_TS7_original(feature_df, params):
    pred_df = []
    for stype, group in feature_df.groupby('Site_type'):
        group['score'] = 0
        for row in params.iterrows():
            if row[0] == 'Intercept':
                group['score'] += row[1]['{} coeff'.format(stype)]
            else:
                feat_min, feat_max, feat_coeff = row[1][['{} min'.format(stype), '{} max'.format(stype), '{} coeff'.format(stype)]]
                vals = group[row[0]]
                group['score'] += feat_coeff * (vals - feat_min) / (feat_max - feat_min)
        pred_df.append(group)
    pred_df = pd.concat(pred_df)
    pred_df['bounded_score'] = np.minimum(pred_df['upper_bound'], pred_df['score'])
    pred_df = pred_df.groupby(['transcript', 'mir']).agg({'score': np.sum, 'bounded_score': np.sum})
    return pred_df
    

    

