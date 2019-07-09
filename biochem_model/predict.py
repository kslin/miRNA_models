import numpy as np
import pandas as pd


def sigmoid(vals):
    return 1 / (1 + np.exp(-1 * vals))


def process_features(filename):

    mean_logSA_diff = 0.85988

    # read in data
    features = pd.read_csv(filename, sep='\t')
    features['log_KA'] = -1 * features['log_kd']
    features['logSA_diff'] = features['logSA_diff'].fillna(mean_logSA_diff)
    features['Threep_canon'] = (features['stype'] != 'no site') * (features['Threep'])
    # features['offset_all1'] = (FITTED_PARAMS['orf_penalty'] * features['in_ORF']) + (FITTED_PARAMS['sa_coef'] * features['logSA_diff'])
    # features['offset1'] = (FITTED_PARAMS['threep_coef'] * features['Threep_canon']) + (FITTED_PARAMS['pct_coef'] * features['PCT'])

    return features


def predict(features, freeAGO, feature_list, kd_cutoff=0, ka_background=0, overwrite_params=None):
    """Predict logFC values from feature dataframe"""

    # Fitted parameters
    FITTED_PARAMS = {
        'log_decay': 0.04719084,
        'orf_penalty': -1.9368109,
        'sa_coef': 0.18783103,
        'threep_coef': 0.26832232,
        'pct_coef': 1.6832638,
    }

    feature_list = feature_list.split(',')
    print(feature_list)
    features = features[features['log_kd'] < kd_cutoff]
    features['offset'] = 0
    features['offset_all'] = FITTED_PARAMS['orf_penalty'] * features['in_ORF']

    if overwrite_params is not None:
        for key, val in overwrite_params.items():
            FITTED_PARAMS[key] = val


    if 'SA' in feature_list:
        features['offset_all'] += FITTED_PARAMS['sa_coef'] * features['logSA_diff']
    if 'Threep_canon' in feature_list:
        features['offset'] += FITTED_PARAMS['threep_coef'] * features['Threep_canon']
    if 'PCT' in feature_list:
        features['offset'] += FITTED_PARAMS['pct_coef'] * features['PCT']
    
    features['occ'] = sigmoid(freeAGO + features['log_KA'] + features['offset'] + features['offset_all'])
    features['occ_init'] = sigmoid(freeAGO + ka_background + features['offset_all'])

    decay = np.exp(FITTED_PARAMS['log_decay'])

    pred = features.groupby(['transcript', 'mir']).agg({'occ': np.sum, 'occ_init': np.sum})
    pred['pred'] = np.log1p(decay * pred['occ_init']) - np.log1p(decay * pred['occ'])
    return features, pred


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
    

    

