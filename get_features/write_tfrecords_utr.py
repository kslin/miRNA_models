from optparse import OptionParser

import numpy as np
import pandas as pd
import tensorflow as tf

import utils
import tf_utils

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_file", dest="TPM_FILE", help="file with TPM data")
    parser.add_option("--feature_file", dest="FEATURE_FILE", help="file with features")
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--passenger", dest="PASSENGER", help="use passenger strand", default=False, action='store_true')
    parser.add_option("--outfile", dest="OUTFILE", help="location for tfrecords")

    (options, args) = parser.parse_args()

    # read miRNA DATA and filter for ones to keep
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
    print("Using mirs: {}".format(ALL_MIRS))

    # read in features
    ALL_FEATS = []
    for mir in ALL_MIRS:
        mir = mir.replace('*', '_pass')
        temp = pd.read_csv(options.FEATURE_FILE.replace('MIR', mir), sep='\t')

        # fill in SA_bg for noncanon sites
        mean_SA_diff = np.nanmean(temp['logSA_diff'])
        temp['logSA_diff'] = temp['logSA_diff'].fillna(mean_SA_diff)

        ALL_FEATS.append(temp)

    ALL_FEATS = pd.concat(ALL_FEATS, sort=False)
    print(len(ALL_FEATS))

    # get rid of noncanonical sites in ORF
    ALL_FEATS = ALL_FEATS[(ALL_FEATS['stype'] != 'no site') | (~ALL_FEATS['in_ORF'])]
    print(len(ALL_FEATS))

    # only take 3p-pairing scores for canonical sites
    ALL_FEATS['Threep_canon'] = ALL_FEATS['Threep'] * (ALL_FEATS['stype'] != 'no site')

    # read in expression data
    TPM = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0)
    for mir in ALL_GUIDES:
        if mir not in TPM.columns:
            raise ValueError('{} given in mirseqs file but not in TPM file.'.format(mir))

    num_batches = 11
    TPM['batch'] = [ix % num_batches for ix in TPM['ix']]
    keep_cols = ['in_ORF', 'logSA_diff', 'Threep_canon', 'PCT']

    with tf.python_io.TFRecordWriter(options.OUTFILE) as tfwriter:
        for ix, row in enumerate(TPM.iterrows()):

            # print progress
            if ix % 100 == 0:
                print("Processed {}/{} transcripts".format(ix, len(TPM)))

            transcript = row[0]
            feat_temp = ALL_FEATS[ALL_FEATS['transcript'] == transcript]

            feature_dict = {
                'transcript': tf_utils._bytes_feature(transcript.encode('utf-8')),
                'batch': tf_utils._int64_feature([row[1]['batch']]),
                'utr3_length': tf_utils._float_feature([row[1]['utr3_length']]),
                'orf_length': tf_utils._float_feature([row[1]['orf_length']]),
            }

            for guide in ALL_GUIDES:
                feature_dict['{}_tpm'.format(guide)] = tf_utils._float_feature([row[1][guide]])

            for mir in ALL_MIRS:

                site8 = MIR_DICT[mir]['site8']
                mirseq = MIR_DICT[mir]['mirseq']

                feature_dict['{}_mir_1hot'.format(mir)] = tf_utils._float_feature(utils.one_hot_encode(mirseq[:options.MIRLEN]))

                feat_mir_temp = feat_temp[feat_temp['mir'] == mir].sort_values('loc')

                nsites = len(feat_mir_temp)
                if nsites > 0:
                    keep_seqs = feat_mir_temp['12mer'].values
                    long_seq = ''.join(keep_seqs)

                    feature_dict['{}_seqs_1hot'.format(mir)] = tf_utils._float_feature(utils.one_hot_encode(long_seq))
                    feature_dict['{}_ts7_features'.format(mir)] = tf_utils._float_feature(list(feat_mir_temp[keep_cols].values.flatten()))

                else:
                    feature_dict['{}_seqs_1hot'.format(mir)] = tf_utils._float_feature([])
                    feature_dict['{}_ts7_features'.format(mir)] = tf_utils._float_feature([])
                    nsites = 0

                feature_dict['{}_nsites'.format(mir)] = tf_utils._int64_feature([nsites])

            # Create a Features message using tf.train.Example.
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            example_proto = example_proto.SerializeToString()

            tfwriter.write(example_proto)
