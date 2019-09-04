import bisect

import numpy as np
import pandas as pd


# Fitted parameters from biochemical model
FITTED_PARAMS = {
    'log_decay': 0.0479742,
    'orf_penalty': -1.936273,
    'sa_coef': 0.1883839,
    'threep_coef': 0.27043834,
    'pct_coef': 1.684825,
}


def sigmoid(vals):
    return 1 / (1 + np.exp(-1 * vals))


def process_features(filename, transcripts=None):

    mean_logSA_diff = 0.898

    # read in data
    features = pd.read_csv(filename, sep='\t')
    if transcripts is not None:
        features = features[features['transcript'].isin(transcripts)]
    features['logSA_diff'] = features['logSA_diff'].fillna(mean_logSA_diff)
    features['Threep_canon'] = (features['stype'] != 'no site') * (features['Threep'])
    # features['offset_all1'] = (FITTED_PARAMS['orf_penalty'] * features['in_ORF']) + (FITTED_PARAMS['sa_coef'] * features['logSA_diff'])
    # features['offset1'] = (FITTED_PARAMS['threep_coef'] * features['Threep_canon']) + (FITTED_PARAMS['pct_coef'] * features['PCT'])

    return features


def predict(features, freeAGO, feature_list, kd_cutoff=10, ka_background=0, overwrite_params=None):
    """Predict logFC values from feature dataframe"""

    features_copy = features.copy()
    feature_list = feature_list.split(',')
    features_copy['log_KA'] = -1 * features_copy['log_kd']
    features_copy = features_copy[features_copy['log_kd'] < kd_cutoff]
    features_copy['offset'] = 0
    features_copy['offset_all'] = FITTED_PARAMS['orf_penalty'] * features_copy['in_ORF']

    if overwrite_params is not None:
        for key, val in overwrite_params.items():
            FITTED_PARAMS[key] = val


    if 'SA' in feature_list:
        features_copy['offset_all'] += FITTED_PARAMS['sa_coef'] * features_copy['logSA_diff']
    if 'Threep_canon' in feature_list:
        features_copy['offset'] += FITTED_PARAMS['threep_coef'] * features_copy['Threep_canon']
    if 'PCT' in feature_list:
        features_copy['offset'] += FITTED_PARAMS['pct_coef'] * features_copy['PCT']
    
    features_copy['occ'] = sigmoid(freeAGO + features_copy['log_KA'] + features_copy['offset'] + features_copy['offset_all'])
    features_copy['occ_init'] = sigmoid(freeAGO + ka_background + features_copy['offset_all'])

    decay = np.exp(FITTED_PARAMS['log_decay'])

    pred = features_copy.groupby(['transcript', 'mir']).agg({'occ': np.sum, 'occ_init': np.sum})
    pred['pred'] = np.log1p(decay * pred['occ_init']) - np.log1p(decay * pred['occ'])
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


def predict_withAIR(features_with_bin, freeAGO, feature_list, AIRS, kd_cutoff=10, ka_background=0, overwrite_params=None):
    """Predict logFC values from feature dataframe with alternative isoform ratio information"""

    DECAY = np.exp(FITTED_PARAMS['log_decay'])

    temp = features_with_bin.copy()
    temp, _ = predict(temp, freeAGO, feature_list, kd_cutoff, ka_background)
    temp = temp[['transcript','bin','occ','occ_init']].groupby(['transcript', 'bin']).agg(np.sum)
    unique_transcripts = features_with_bin['transcript'].unique()

    temp2 = AIRS[AIRS['transcript'].isin(unique_transcripts)].set_index(['transcript', 'bin']).sort_index()
    temp2 = pd.concat([temp, temp2], axis=1, join_axes=[temp2.index]).reset_index().fillna(0)

    results = []
    for transcript, group in temp2.groupby('transcript'):
        group['occ'] = np.cumsum(group['occ'])
        group['occ_init'] = np.cumsum(group['occ_init'])
        group['pred'] = np.log1p(DECAY*group['occ_init']) - np.log1p(DECAY*group['occ'])
        if np.abs(np.sum(group['percent_end_here']) - 1) > 0.0001:
            print(np.sum(group['percent_end_here']))
            raise ValueError()
        results.append([transcript, np.sum(group['pred'] * group['percent_end_here'])])

    results = pd.DataFrame(results, columns=['transcript', f'pred_{freeAGO}']).set_index('transcript')
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
    

    

