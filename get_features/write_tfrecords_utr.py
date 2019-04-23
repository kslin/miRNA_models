from optparse import OptionParser
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import get_site_features
import utils

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    parser.add_option("-m", "--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--pct", dest="PCT_FILE", help="file with PCTs")
    parser.add_option("--orf_file", dest="ORF_FILE", help="ORF sequences in tsv format")
    parser.add_option("-r", "--rnaplfold_folder", dest="RNAPLFOLD_FOLDER", help="location of RNAPLfold lunp files")
    parser.add_option("-w", "--outfile", dest="OUTFILE", help="location for tfrecords")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--overlap_dist", dest="OVERLAP_DIST", help="minimum distance between neighboring sites", type=int)
    parser.add_option("--upstream_limit", dest="UPSTREAM_LIMIT", help="how far upstream to look for 3p pairing", type=int)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--only_canon", dest="ONLY_CANON", help="only use canonical sites", default=False, action='store_true')
    parser.add_option("--write_seqs", dest="WRITE_SEQS", help="write seqs found", default=None)
    parser.add_option("--calc_ts7", dest="CALC_TS7", help="calculate TS7 features", default=False, action='store_true')

    (options, args) = parser.parse_args()

    ### READ miRNA DATA and filter for ones to keep ###
    MIRNAS = pd.read_csv(options.MIR_SEQS, sep='\t')
    MIRNAS = MIRNAS[MIRNAS['use_tpms']]
    ALL_GUIDES = sorted(list(MIRNAS['mir'].values))

    MIR_DICT = {}
    ALL_MIRS = []
    for row in MIRNAS.iterrows():
        guide_seq = row[1]['guide_seq']
        pass_seq = row[1]['pass_seq']
        MIR_DICT[row[1]['mir']] = {
            'mirseq': guide_seq,
            'site8': utils.rev_comp(guide_seq[1:8]) + 'A',
            'one_hot': utils.one_hot_encode(guide_seq[:options.MIRLEN])
        }
        ALL_MIRS.append(row[1]['mir'])
        if options.PASSENGER:
            MIR_DICT[row[1]['mir'] + '*'] = {
                'mirseq': pass_seq,
                'site8': utils.rev_comp(pass_seq[1:8]) + 'A',
                'one_hot': utils.one_hot_encode(pass_seq[:options.MIRLEN])
            }
            ALL_MIRS.append(row[1]['mir'] + '*')

    ALL_MIRS = sorted(ALL_MIRS)

    ### READ EXPRESSION DATA ###
    TPM = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)
    for mir in ALL_GUIDES:
        if mir not in TPM.columns:
            raise ValueError('{} given in mirseqs file but not in TPM file.'.format(mir))

    num_batches = 11
    TPM['batch'] = [ix % num_batches for ix in TPM['ix']]

    print(TPM['batch'].unique())

    print("Using mirs: {}".format(ALL_MIRS))

    pct_df = pd.read_csv(options.PCT_FILE, sep='\t', usecols=['Gene ID', 'miRNA family', 'Site type', 'Site start', 'PCT'])
    pct_df['offset'] = [1 if x in ['8mer-1a','7mer-m8'] else 0 for x in pct_df['Site type']]
    pct_df['Site start'] = pct_df['Site start'] + pct_df['offset']
    pct_df['ID'] = pct_df['Gene ID'] + pct_df['miRNA family']
    pct_df = pct_df[['ID', 'Site start', 'PCT']].set_index('Site start')

    # read in orf sequences
    ORF_SEQS = pd.read_csv(options.ORF_FILE, sep='\t', header=None, index_col=0)

    if options.WRITE_SEQS is not None:
        seq_writer = open(options.WRITE_SEQS, 'w')

    NUM_FEATURES = 8

    with tf.python_io.TFRecordWriter(options.OUTFILE) as tfwriter:
        for ix, row in enumerate(TPM.iterrows()):

            # print progress
            if ix % 100 == 0:
                print("Processed {}/{} transcripts".format(ix, len(TPM)))

            transcript = row[0]
            utr3 = row[1]['sequence']
            orf = ORF_SEQS.loc[transcript][2]
            # transcript_sequence = orf + utr3
            # orf_length = len(orf)
            # transcript_length = len(transcript_sequence)

            lunp_file = os.path.join(options.RNAPLFOLD_FOLDER, transcript) + '_lunp'
            rnaplfold_data = pd.read_csv(lunp_file, sep='\t', header=1).set_index(' #i$').astype(float)

            utr_len = row[1]['utr_length']
            orf_len = row[1]['orf_length']

            feature_dict = {
                'transcript': utils._bytes_feature(transcript.encode('utf-8')),
                'batch': utils._int64_feature([row[1]['batch']]),
            }

            for guide in ALL_GUIDES:
                feature_dict['{}_tpm'.format(guide)] = utils._float_feature([row[1][guide]])

            for mir in ALL_MIRS:

                site8 = MIR_DICT[mir]['site8']
                mirseq = MIR_DICT[mir]['mirseq']

                feature_dict['{}_mir_1hot'.format(mir)] = utils._float_feature(utils.one_hot_encode(mirseq[:options.MIRLEN]))

                # get sites and 12mer sequences
                orf_seqs, orf_locs = get_site_features.get_sites_from_utr(orf, site8,
                    overlap_dist=options.OVERLAP_DIST, only_canon=True)

                utr3_seqs, utr3_locs = get_site_features.get_sites_from_utr(utr3, site8,
                    overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)

                all_seqs = orf_seqs + utr3_seqs
                nsites = len(all_seqs)

                if nsites > 0:
                    if options.CALC_TS7:
                        orf_features = get_site_features.get_ts7_features(mirseq, orf_locs, orf_locs, orf, utr_len, orf_len,
                                                                          options.UPSTREAM_LIMIT, None, None, in_orf=True).flatten()

                        utr3_stypes = [utils.get_centered_stype(site8, seq) for seq in utr3_seqs]
                        pct_df_temp = pct_df[pct_df['ID'] == transcript + mir]
                        if len(pct_df_temp) == 0:
                            pct_df_temp = None
                        utr3_features = get_site_features.get_ts7_features(mirseq, utr3_locs, utr3_stypes, utr3, utr_len, orf_len,
                                                                          options.UPSTREAM_LIMIT, rnaplfold_data,
                                                                          pct_df_temp, in_orf=False).flatten()

                        if int(len(utr3_features) / NUM_FEATURES) != len(utr3_locs):
                            raise ValueError('Number of features do not match')

                        features = list(orf_features) + list(utr3_features)
                    else:
                        features = ([1.0] * len(orf_seqs)) + ([0.0] * len(utr3_seqs))
                    long_seq = ''.join(all_seqs)

                    if options.WRITE_SEQS is not None:
                        seq_writer.write('{}\t{}\t{}\t{}\n'.format(transcript, mir, mirseq, ','.join(all_seqs)))

                    feature_dict['{}_seqs_1hot'.format(mir)] = utils._float_feature(utils.one_hot_encode(long_seq))
                    feature_dict['{}_ts7_features'.format(mir)] = utils._float_feature(features)


                    # features = []
                    # for l in locs:
                    #     features += [float(l < orf_length), float((transcript_length - l) < 500)]

                    # # features = [1.0 if (l < orf_length) else 0.0 for l in locs]

                    # # stypes = [utils.get_centered_stype(site8, seq) for seq in seqs]
                    # # pct_df_temp = pct_df[pct_df['ID'] == transcript + mir]
                    # # if len(pct_df_temp) == 0:
                    # #     pct_df_temp = None
                    # # features = get_site_features.get_ts7_features(mirseq, locs, stypes, utr, utr_len, orf_len,
                    # #                                                   options.UPSTREAM_LIMIT, rnaplfold_data,
                    # #                                                   pct_df_temp).flatten()
                    # # print(features)
                    # feature_dict['{}_ts7_features'.format(mir)] = utils._float_feature(features)

                else:
                    feature_dict['{}_seqs_1hot'.format(mir)] = utils._float_feature([])
                    feature_dict['{}_ts7_features'.format(mir)] = utils._float_feature([])
                    nsites = 0

                feature_dict['{}_nsites'.format(mir)] = utils._int64_feature([nsites])

            # Create a Features message using tf.train.Example.
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            example_proto = example_proto.SerializeToString()

            tfwriter.write(example_proto)
            # break

