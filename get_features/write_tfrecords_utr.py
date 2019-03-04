from optparse import OptionParser
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import utils

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    parser.add_option("-m", "--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--pct", dest="PCT_FILE", help="file with PCTs")
    # parser.add_option("--orf_file", dest="ORF_FILE", help="ORF sequences in tsv format")
    parser.add_option("-r", "--rnaplfold_folder", dest="RNAPLFOLD_FOLDER", help="location of RNAPLfold lunp files")
    parser.add_option("-w", "--outfile", dest="OUTFILE", help="location for tfrecords")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--overlap_dist", dest="OVERLAP_DIST", help="minimum distance between neighboring sites", type=int)
    parser.add_option("--upstream_limit", dest="UPSTREAM_LIMIT", help="how far upstream to look for 3p pairing", type=int)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--only_canon", dest="ONLY_CANON", help="only use canonical sites", default=False, action='store_true')
    parser.add_option("--write_seqs", dest="WRITE_SEQS", help="write seqs found", default=None)

    (options, args) = parser.parse_args()

    ### READ miRNA DATA and filter for ones to keep ###
    MIRNAS = pd.read_csv(options.MIR_SEQS, sep='\t')
    MIRNAS = MIRNAS[MIRNAS['use_tpms']]
    ALL_GUIDES = list(MIRNAS['mir'].values)

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

    ### READ EXPRESSION DATA ###
    TPM = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)
    for mir in ALL_GUIDES:
        if mir not in TPM.columns:
            raise ValueError('{} given in mirseqs file but not in TPM file.'.format(mir))

    print("Using mirs: {}".format(ALL_MIRS))

    pct_df = pd.read_csv(options.PCT_FILE, sep='\t', usecols=['Gene ID', 'miRNA family', 'Site type', 'Site start', 'PCT'])
    pct_df['offset'] = [1 if x in ['8mer-1a','7mer-m8'] else 0 for x in pct_df['Site type']]
    pct_df['Site start'] = pct_df['Site start'] + pct_df['offset']
    pct_df['ID'] = pct_df['Gene ID'] + pct_df['miRNA family']
    pct_df = pct_df[['ID', 'Site start', 'PCT']].set_index('Site start')

    # # read in orf sequences
    # ORF_SEQS = pd.read_csv(options.ORF_FILE, sep='\t', header=None, index_col=0)

    if options.WRITE_SEQS is not None:
        seq_writer = open(options.WRITE_SEQS, 'w')

    with tf.python_io.TFRecordWriter(options.OUTFILE) as tfwriter:
        for ix, row in enumerate(TPM.iterrows()):

            # print progress
            if ix % 100 == 0:
                print("Processed {}/{} transcripts".format(ix, len(TPM)))

            transcript = row[0]
            utr = row[1]['sequence']

            lunp_file = os.path.join(options.RNAPLFOLD_FOLDER, transcript) + '_lunp'
            rnaplfold_data = pd.read_csv(lunp_file, sep='\t', header=1).set_index(' #i$').astype(float)

            utr_len = row[1]['utr_length']
            orf_len = row[1]['orf_length']

            feature_dict = {
                'transcript': utils._bytes_feature(transcript.encode('utf-8')),
                'tpms': utils._float_feature(list(row[1][ALL_GUIDES].values)),
            }

            nsites = []

            for mir in ALL_MIRS:

                site8 = MIR_DICT[mir]['site8']
                mirseq = MIR_DICT[mir]['mirseq']

                feature_dict['{}_mir_1hot'.format(mir)] = utils._float_feature(utils.one_hot_encode(mirseq[:options.MIRLEN]))

                # get sites and 12mer sequences
                seqs, locs = utils.get_sites_from_utr(utr, site8, overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)
                nsites.append(len(locs))

                if len(locs) > 0:
                    if options.WRITE_SEQS is not None:
                        seq_writer.write('{}\t{}\t{}\t{}\n'.format(transcript, mir, mirseq, ','.join(seqs)))
                    long_seq = ''.join(seqs)
                    feature_dict['{}_seqs_1hot'.format(mir)] = utils._float_feature(utils.one_hot_encode(long_seq))

                    features = utils.get_ts7_features(mirseq, locs, utr, utr_len, orf_len, options.UPSTREAM_LIMIT, rnaplfold_data)
                    # print(features)
                    feature_dict['{}_ts7_features'.format(mir)] = utils._float_feature(features)

                else:
                    feature_dict['{}_seqs_1hot'.format(mir)] = utils._float_feature([])
                    feature_dict['{}_ts7_features'.format(mir)] = utils._float_feature([])

            feature_dict['nsites'] = utils._int64_feature(nsites)

            if np.sum(nsites) == 0:
                print('Skipping {} because no sites found'.format(transcript))
            else:
                # Create a Features message using tf.train.Example.
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                example_proto = example_proto.SerializeToString()

                tfwriter.write(example_proto)
            # break

