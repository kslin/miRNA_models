from optparse import OptionParser
import os
import sys
import time

import numpy as np
import pandas as pd

import utils
import get_features

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_file", dest="TPM_FILE", help="tpm data")
    parser.add_option("--kd_file", dest="KD_FILE", help="kd data")
    parser.add_option("--orf_file", dest="ORF_FILE", help="ORF sequences in tsv format")
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--pct", dest="PCT_FILE", help="file with PCTs")
    parser.add_option("--ta_sps", dest="TA_SPS_FILE", help="file with TA and SPS")
    parser.add_option("--rnaplfold_folder", dest="RNAPLFOLD_FOLDER", help="location of RNAPLfold lunp files")
    parser.add_option("--outdir", dest="OUTDIR", help="location for writing output files")
    parser.add_option("--overlap_dist", dest="OVERLAP_DIST", help="minimum distance between neighboring sites", type=int)
    parser.add_option("--upstream_limit", dest="UPSTREAM_LIMIT", help="how far upstream to look for 3p pairing", type=int)
    parser.add_option("--only_canon", dest="ONLY_CANON", help="only use canonical sites", default=False, action='store_true')
    parser.add_option("--only_rbns", dest="ONLY_RBNS", help="only use miRNAs with RBNS data", default=False, action='store_true')

    (options, args) = parser.parse_args()

    if (not os.path.isdir(options.OUTDIR)):
        os.makedirs(options.OUTDIR)

    ### READ miRNA DATA and filter for ones to keep ###
    MIRNAS = pd.read_csv(options.MIR_SEQS, sep='\t').sort_values('mir')
    MIRNAS = MIRNAS[MIRNAS['use_tpms']]
    if options.ONLY_RBNS:
        MIRNAS = MIRNAS[MIRNAS['has_rbns']]
    ALL_GUIDES = list(MIRNAS['mir'].values)

    MIR_DICT = {}
    ALL_MIRS = []
    for row in MIRNAS.iterrows():
        guide_seq = row[1]['guide_seq']
        MIR_DICT[row[1]['mir']] = {
            'mirseq': guide_seq,
            'site8': utils.rev_comp(guide_seq[1:8]) + 'A'
        }
        ALL_MIRS.append(row[1]['mir'])

    ### READ EXPRESSION DATA ###
    TPM = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0).sort_index()
    for mir in ALL_GUIDES:
        if mir not in TPM.columns:
            raise ValueError('{} given in mirseqs file but not in TPM file.'.format(mir))

    print("Using mirs: {}".format(ALL_MIRS))

    # read in orf sequences
    ORF_SEQS = pd.read_csv(options.ORF_FILE, sep='\t', header=None, index_col=0)

    col_order = ['transcript', 'transcript_ix', 'mir', 'mirseq', '6mer_loc', 'seq', 'stype'] + \
                ['log_KA', 'TA', 'SPS', 'Local_AU', 'Threep', 'Min_dist', 'SA', 'UTR_len', 'ORF_len', 'PCT', 'Off6m', 'ORF_8m'] + \
                ['siRNA_1', 'siRNA_8', 'site_8']

    col_order_orf_sites = ['transcript', 'transcript_ix', 'mir', 'stype', 'log_KA']

    with open(os.path.join(options.OUTDIR, 'features.txt'), 'w') as outfile:
        outfile.write('\t'.join(col_order) + '\n')

    with open(os.path.join(options.OUTDIR, 'orf_sites.txt'), 'w') as outfile:
        outfile.write('\t'.join(col_order_orf_sites) + '\n')

    pct_df = pd.read_csv(options.PCT_FILE, sep='\t', usecols=['Gene ID', 'miRNA family', 'Site type', 'Site start', 'PCT'])
    pct_df['offset'] = [1 if x in ['8mer-1a','7mer-m8'] else 0 for x in pct_df['Site type']]
    pct_df['Site start'] = pct_df['Site start'] + pct_df['offset']
    pct_df['ID'] = pct_df['Gene ID'] + pct_df['miRNA family']
    pct_df = pct_df[['ID', 'Site start', 'PCT']].set_index('Site start')

    ta_sps_df = pd.read_csv(options.TA_SPS_FILE, sep='\t', index_col=0)

    kd_df = pd.read_csv(options.KD_FILE, sep='\t')
    kd_dict = {}
    for mir, group in kd_df.groupby('mir'):
        kd_dict[mir] = group.set_index('12mer')

    with open(os.path.join(options.OUTDIR, 'features.txt'), 'a') as outfile:
        with open(os.path.join(options.OUTDIR, 'orf_sites.txt'), 'a') as orf_sites_outfile:
            for ix, row in enumerate(TPM.iterrows()):

                # print progress
                if ix % 100 == 0:
                    print("Processed {}/{} transcripts".format(ix, len(TPM)))

                transcript = row[0]
                utr = row[1]['sequence']
                utrU = utr.replace('T', 'U')
                orf = ORF_SEQS.loc[transcript][2]

                lunp_file = os.path.join(options.RNAPLFOLD_FOLDER, transcript) + '_lunp'
                rnaplfold_data = pd.read_csv(lunp_file, sep='\t', header=1).set_index(' #i$').astype(float)

                utr_len = row[1]['utr_length']
                orf_len = row[1]['orf_length']

                assert orf_len == len(orf)

                feature_df = []
                orf_sites_df = []

                for mir in ALL_MIRS:

                    site8 = MIR_DICT[mir]['site8']
                    mirseq = MIR_DICT[mir]['mirseq']
                    mirseqU = mirseq.replace('T','U')
                    seed_region = mirseq[1:8]

                    # get sites and 12mer sequences from UTR
                    seqs, locs = get_features.get_sites_from_utr(utr, site8, overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)

                    if len(locs) > 0:
                        features = get_features.get_ts7_features(mirseq, locs, utr, utr_len, orf_len, options.UPSTREAM_LIMIT, rnaplfold_data)
                        if mir in kd_dict:
                            features['log_KA'] = -1 * kd_dict[mir].loc[seqs]['log_kd'].values
                        else:
                            features['log_KA'] = np.nan
                        features['transcript'] = transcript
                        features['transcript_ix'] = ix
                        features['mir'] = mir
                        features['mirseq'] = mirseq
                        features['seq'] = seqs
                        features['6mer_loc'] = locs
                        # features['best_stype'] = [utils.get_best_stype(site8, seq) for seq in seqs]
                        stypes = [utils.get_centered_stype(site8, seq) for seq in seqs]
                        features['stype'] = stypes
                        features['Off6m'] = stypes.count('6mer-m8')
                        features['siRNA_1'] = mirseq[0]
                        features['siRNA_8'] = mirseq[7]
                        features['site_8'] = [seq[9] for seq in seqs]
                        features['SPS'] = ta_sps_df.loc[seed_region][stypes].values
                        features['TA'] = ta_sps_df.loc[seed_region]['TA']
                        features['ORF_8m'] = orf.count(site8)

                        # add Namita's 3p score
                        # features['Threep_NB'] = [utils.threepscore_NB(mirseqU, utrU, min(utr_len - 1, loc + 7))[0] for loc in locs]

                        pcts = []
                        pct_df_temp = pct_df[pct_df['ID'] == transcript + mir]
                        try:
                            for (x,y) in zip(features['6mer_loc'], features['stype']):
                                if y in ['6mer', '7mer-a1', '7mer-m8', '8mer']:
                                    pcts.append(pct_df_temp.loc[x]['PCT'])
                                else:
                                    pcts.append(0.0)

                        except:
                            print(pct_df_temp)
                            print(features[['6mer_loc','stype']])
                            raise ValueError('locations do not match for {}'.format(transcript))

                        features['PCT'] = pcts
                        feature_df.append(features)

                    # get sites and 12mer sequences for ORF sites, only canonical sites
                    seqs, locs = get_features.get_sites_from_utr(orf, site8, overlap_dist=options.OVERLAP_DIST, only_canon=True)

                    if len(locs) > 0:
                        features = pd.DataFrame({
                            'transcript': [transcript] * len(locs),
                            'transcript_ix': [ix] * len(locs),
                            'mir': [mir] * len(locs),
                        })
                        stypes = [utils.get_centered_stype(site8, seq) for seq in seqs]
                        features['stype'] = stypes
                        if mir in kd_dict:
                            features['log_KA'] = -1 * kd_dict[mir].loc[seqs]['log_kd'].values
                        else:
                            features['log_KA'] = np.nan

                        orf_sites_df.append(features)

                if len(feature_df) > 0:
                    feature_df = pd.concat(feature_df, sort=True)[col_order]
                    feature_df.to_csv(outfile, sep='\t', header=False, index=False)

                if len(orf_sites_df) > 0:
                    orf_sites_df = pd.concat(orf_sites_df, sort=True)[col_order_orf_sites]
                    orf_sites_df.to_csv(orf_sites_outfile, sep='\t', header=False, index=False)
