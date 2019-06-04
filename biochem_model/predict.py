from optparse import OptionParser
import os
import sys
import time

import numpy as np
import pandas as pd

import utils

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--transcripts", dest="TRANSCRIPTS", help="transcript sequence information")
    parser.add_option("--kds", dest="KDS", help="kd data in tsv format")
    parser.add_option("--sa_bg", dest="SA_BG", help="SA background for 12mers")
    parser.add_option("--mirs", dest="MIRS", help="miRNAs to predict")
    parser.add_option("--rnaplfold_dir", dest="RNAPLFOLD_DIR", help="folder with RNAplfold info for transcripts")
    parser.add_option("--kd_cutoff", dest="KD_CUTOFF", type=float)
    parser.add_option("--outfile", dest="OUTFILE", help="location to write outputs")

    (options, args) = parser.parse_args()

    TRANSCRIPTS = pd.read_csv(options.TRANSCRIPTS, sep='\t', index_col='transcript')

    KDS = pd.read_csv(options.KDS, sep='\t')
    KDS = KDS[KDS['best_stype'] == KDS['aligned_stype']]

    SA_BG = pd.read_csv(options.SA_BG, sep='\t', index_col='12mer')

    MIRS = options.MIRS.split(',')

    mir_kd_dict = {}
    for mir in MIRS:
        temp = KDS[KDS['mir'] == mir]
        if len(temp) == 0:
            raise ValueError('{} not in kd and sa_bg files'.format(mir))
        seq_dict_temp = {x:y for (x,y) in zip(temp['12mer'], temp['log_kd']) if (y < options.KD_CUTOFF)}
        mir_kd_dict[mir] = seq_dict_temp

    # find all the sites and KDs
    all_sites = []
    for row in TRANSCRIPTS.iterrows():
        all_sites.append(utils.get_sites_no_overlap(row[0], row[1]['orf_utr3'], MIRS, mir_kd_dict))
    all_sites = pd.concat(all_sites)

    # add site accessibility background information
    all_features = []
    for mir, group in all_sites.groupby('mir'):
        temp = SA_BG[SA_BG['mir'] == mir]
        group['logSA_bg'] = temp.reindex(group['12mer'].values)['logp'].values

        temp = KDS[KDS['mir'] == mir].set_index('12mer')
        group['stype'] = temp.reindex(group['12mer'].values)['aligned_stype'].values
        all_features.append(group)

    temp = pd.concat(all_features).sort_values(['transcript', 'mir'])

    # add site accessibility information
    all_features = []
    for transcript, group in temp.groupby('transcript'):
        lunp_file = os.path.join(options.RNAPLFOLD_DIR, transcript) + '_lunp'
        rnaplfold_data = pd.read_csv(lunp_file, sep='\t', header=1).set_index(' #i$').astype(float)
        locs = group['loc'].values
        group['SA'] = rnaplfold_data.reindex(locs + 7)['14'].values.astype(float) # Agarwal 2015 parameters
        group['logSA'] = np.log(group['SA'])
        all_features.append(group)

    all_features = pd.concat(all_features).sort_values(['transcript', 'mir'])
    all_features['orf_length'] = TRANSCRIPTS.reindex(all_features['transcript'].values)['orf_length'].values
    all_features['utr3_length'] = TRANSCRIPTS.reindex(all_features['transcript'].values)['utr3_length'].values
    all_features['in_ORF'] = all_features['loc'] < (all_features['orf_length'] + 15)
    all_features['log_KA'] = -1 * all_features['log_kd']
    
    all_features['logSA_diff'] = all_features['logSA'] - all_features['logSA_bg']
    mean_SA_diff = np.mean(all_features.dropna()['logSA_diff'])
    all_features['logSA_diff'] = all_features['logSA_diff'].fillna(mean_SA_diff)
    all_features['logSA_bg'] = all_features['logSA'] - all_features['logSA_diff']

    # write outputs
    all_features.to_csv(options.OUTFILE, sep='\t')

