from optparse import OptionParser
import os

import numpy as np
import pandas as pd

import get_site_features
import utils

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--transcripts", dest="TRANSCRIPTS", help="transcript sequence information")
    parser.add_option("--mir", dest="MIR", help="miRNA to get features for")
    parser.add_option("-m", "--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--kds", dest="KDS", help="kd data in tsv format", default=None)
    parser.add_option("--sa_bg", dest="SA_BG", help="SA background for 12mers")
    parser.add_option("--rnaplfold_dir", dest="RNAPLFOLD_DIR", help="folder with RNAplfold info for transcripts")
    parser.add_option("--pct_file", dest="PCT_FILE", default=None, help="file with PCT information")
    parser.add_option("--kd_cutoff", dest="KD_CUTOFF", type=float, default=None)
    parser.add_option("--outfile", dest="OUTFILE", help="location to write outputs")
    parser.add_option("--overlap_dist", dest="OVERLAP_DIST", help="minimum distance between neighboring sites", type=int)
    parser.add_option("--upstream_limit", dest="UPSTREAM_LIMIT", help="how far upstream to look for 3p pairing", type=int)
    parser.add_option("--only_canon", dest="ONLY_CANON", help="only use canonical sites", default=False, action='store_true')

    (options, args) = parser.parse_args()

    if options.MIR == 'let7c':
        KD_MIR = 'let7'
        FEATURE_MIR = 'let7c'
    else:
        KD_MIR = options.MIR
        FEATURE_MIR = options.MIR

    TRANSCRIPTS = pd.read_csv(options.TRANSCRIPTS, sep='\t', index_col='transcript')
    mirseqs = pd.read_csv(options.MIR_SEQS, sep='\t', index_col='mir')
    if '_pass' in KD_MIR:
        MIRSEQ = mirseqs.loc[KD_MIR.replace('_pass', '')]['pass_seq']
    else:
        MIRSEQ = mirseqs.loc[KD_MIR]['guide_seq']

    SITE8 = utils.rev_comp(MIRSEQ[1:8]) + 'A'
    print(KD_MIR, FEATURE_MIR, SITE8)

    # if KD file provided, find sites based on KD file
    if options.KDS is not None:
        KDS = pd.read_csv(options.KDS, sep='\t')
        if options.ONLY_CANON:
            KDS = KDS[KDS['aligned_stype'] != 'no site']
        KDS = KDS[KDS['best_stype'] == KDS['aligned_stype']]

        temp = KDS[KDS['mir'] == KD_MIR]
        if len(temp) == 0:
            raise ValueError('{} not in kd files'.format(KD_MIR))
        mir_kd_dict = {x: y for (x, y) in zip(temp['12mer'], temp['log_kd']) if (y < options.KD_CUTOFF)}

        # find all the sites and KDs
        all_features = []
        for row in TRANSCRIPTS.iterrows():
            all_features.append(get_site_features.get_sites_from_kd_dict(row[0], row[1]['orf_utr3'], mir_kd_dict, options.OVERLAP_DIST))

    # otherwise, go by sequence
    else:
        all_features = []
        for row in TRANSCRIPTS.iterrows():
            all_features.append(get_site_features.get_sites_from_sequence(row[0], row[1]['orf_utr3'], SITE8,
                            overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON))

    all_features = pd.concat(all_features).sort_values('transcript')
    all_features['mir'] = FEATURE_MIR.replace('_pass', '*')

    # add site accessibility background information
    temp = pd.read_csv(options.SA_BG.replace('MIR', KD_MIR), sep='\t', index_col='12mer').reindex(all_features['12mer'].values)
    all_features['logSA_bg'] = temp['logp'].values

    # add stypes
    all_features['stype'] = [utils.get_centered_stype(SITE8, seq) for seq in all_features['12mer'].values]

    # sanity check on background
    temp = all_features[all_features['stype'] != 'no site']
    if len(temp) != len(temp.dropna()):
        raise ValueError('Error in site accessibility background assignment')

    print('Adding 3p score and SA')

    # add transcript-specific information
    temp = []
    for transcript, group in all_features.groupby('transcript'):
        locs = group['loc'].values
        
        # add threep pairing score
        sequence = TRANSCRIPTS.loc[transcript]['orf_utr3']
        group['Threep'] = [get_site_features.calculate_threep_score(MIRSEQ, sequence, int(loc - 3), options.UPSTREAM_LIMIT) for loc in locs]

        # add site accessibility information
        lunp_file = os.path.join(options.RNAPLFOLD_DIR, transcript) + '.txt'
        rnaplfold_data = pd.read_csv(lunp_file, sep='\t', index_col='end')
        group['SA'] = rnaplfold_data.reindex(locs + 7)['14'].values.astype(float)  # Agarwal 2015 parameters
        group['logSA'] = np.log(group['SA'])

        temp.append(group)

    all_features = pd.concat(temp)
    all_features['orf_length'] = TRANSCRIPTS.reindex(all_features['transcript'].values)['orf_length'].values
    all_features['utr3_length'] = TRANSCRIPTS.reindex(all_features['transcript'].values)['utr3_length'].values
    all_features['in_ORF'] = all_features['loc'] < (all_features['orf_length'] + 15)
    all_features['logSA_diff'] = all_features['logSA'] - all_features['logSA_bg']
    all_features['utr3_loc'] = all_features['loc'] - all_features['orf_length']

    print('Adding PCT')

    # add PCT information if indicated
    if options.PCT_FILE is not None:
        pct_df = pd.read_csv(options.PCT_FILE, sep='\t', usecols=['Gene ID', 'miRNA family', 'Site type', 'Site start', 'PCT'])
        pct_df.columns = ['transcript', 'mir', 'stype', 'loc', 'PCT']
        pct_df = pct_df[pct_df['mir'] == FEATURE_MIR]
        if len(pct_df) == 0:
            all_features['PCT'] = 0
            print(f"No PCT information for {FEATURE_MIR}")
        else:
            pct_df['offset'] = [1 if x in ['8mer-1a', '7mer-m8'] else 0 for x in pct_df['stype']]
            pct_df['loc'] = pct_df['loc'] + pct_df['offset']
            pct_df = pct_df[pct_df['stype'] != '6mer']
            pct_df = pct_df.set_index(['transcript', 'loc'])

            temp1 = all_features[all_features['in_ORF']]
            temp1['PCT'] = 0
            temp2 = all_features[~all_features['in_ORF']]
            temp2['PCT'] = pct_df.reindex(temp2[['transcript', 'utr3_loc']])['PCT'].values
            temp2['PCT'] = temp2['PCT'].fillna(0.0)
            all_features = pd.concat([temp1, temp2])

    all_features = all_features.set_index('transcript').sort_index()

    # write outputs
    all_features.to_csv(options.OUTFILE.replace('MIR', FEATURE_MIR), sep='\t')
