from optparse import OptionParser
import os
import sys
import time

import numpy as np
import pandas as pd

import utils
import get_site_features

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_file", dest="TPM_FILE", help="tpm data")
    parser.add_option("--kd_file", dest="KD_FILE", help="kd data")
    parser.add_option("--orf_file", dest="ORF_FILE", help="ORF sequences in tsv format")
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--outdir", dest="OUTDIR", help="location for writing output files")
    parser.add_option("--overlap_dist", dest="OVERLAP_DIST", help="minimum distance between neighboring sites", type=int)
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

    col_order = ['transcript', 'transcript_ix', 'mir', 'mirseq', '6mer_loc', 'seq', 'stype', 'log_KA']

    with open(os.path.join(options.OUTDIR, 'utr_sites.txt'), 'w') as outfile:
        outfile.write('\t'.join(col_order) + '\n')

    with open(os.path.join(options.OUTDIR, 'orf_sites.txt'), 'w') as outfile:
        outfile.write('\t'.join(col_order) + '\n')

    kd_df = pd.read_csv(options.KD_FILE, sep='\t')
    kd_dict = {}
    for mir, group in kd_df.groupby('mir'):
        kd_dict[mir] = group.set_index('12mer')

    with open(os.path.join(options.OUTDIR, 'utr_sites.txt'), 'a') as outfile:
        with open(os.path.join(options.OUTDIR, 'orf_sites.txt'), 'a') as orf_sites_outfile:
            for ix, row in enumerate(TPM.iterrows()):

                # print progress
                if ix % 100 == 0:
                    print("Processed {}/{} transcripts".format(ix, len(TPM)))

                transcript = row[0]
                utr = row[1]['sequence']
                orf = ORF_SEQS.loc[transcript][2]

                utr_sites_df = []
                orf_sites_df = []

                for mir in ALL_MIRS:

                    site8 = MIR_DICT[mir]['site8']
                    mirseq = MIR_DICT[mir]['mirseq']

                    # get sites and 12mer sequences from UTR
                    seqs, locs = get_site_features.get_sites_from_utr(utr, site8, overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)

                    if len(locs) > 0:
                        features = pd.DataFrame({
                            'transcript': [transcript] * len(locs),
                            'transcript_ix': [ix] * len(locs),
                            'mir': [mir] * len(locs),
                            'seq': seqs,
                            '6mer_loc': locs
                        })
                        features['mirseq'] = mirseq
                        stypes = [utils.get_centered_stype(site8, seq) for seq in seqs]
                        features['stype'] = stypes
                        if mir in kd_dict:
                            features['log_KA'] = -1 * kd_dict[mir].loc[seqs]['log_kd'].values
                        else:
                            features['log_KA'] = np.nan

                        utr_sites_df.append(features)

                    # get sites and 12mer sequences for ORF sites
                    seqs, locs = get_site_features.get_sites_from_utr(orf, site8, overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)

                    if len(locs) > 0:
                        features = pd.DataFrame({
                            'transcript': [transcript] * len(locs),
                            'transcript_ix': [ix] * len(locs),
                            'mir': [mir] * len(locs),
                            'seq': seqs,
                            '6mer_loc': locs
                        })
                        features['mirseq'] = mirseq
                        stypes = [utils.get_centered_stype(site8, seq) for seq in seqs]
                        features['stype'] = stypes
                        if mir in kd_dict:
                            features['log_KA'] = -1 * kd_dict[mir].loc[seqs]['log_kd'].values
                        else:
                            features['log_KA'] = np.nan

                        orf_sites_df.append(features)

                if len(utr_sites_df) > 0:
                    utr_sites_df = pd.concat(utr_sites_df, sort=True)[col_order]
                    utr_sites_df.to_csv(outfile, sep='\t', header=False, index=False, float_format='%.3f')

                if len(orf_sites_df) > 0:
                    orf_sites_df = pd.concat(orf_sites_df, sort=True)[col_order]
                    orf_sites_df.to_csv(orf_sites_outfile, sep='\t', header=False, index=False, float_format='%.3f')
