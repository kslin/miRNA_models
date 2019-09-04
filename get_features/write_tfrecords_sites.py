from optparse import OptionParser
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import utils
import get_site_features
import tf_utils

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--tpm_file", dest="TPM_FILE", help="tpm data")
    parser.add_option("--orf_file", dest="ORF_FILE", help="ORF sequences in tsv format")
    parser.add_option("--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("-w", "--outfile", dest="OUTFILE", help="location for tfrecords")
    parser.add_option("--overlap_dist", dest="OVERLAP_DIST", help="minimum distance between neighboring sites", type=int)
    parser.add_option("--only_canon", dest="ONLY_CANON", help="only use canonical sites", default=False, action='store_true')

    (options, args) = parser.parse_args()

    ### READ miRNA DATA and filter for ones to keep ###
    MIRNAS = pd.read_csv(options.MIR_SEQS, sep='\t')
    MIRNAS = MIRNAS[MIRNAS['use_tpms']]
    ALL_GUIDES = sorted(list(MIRNAS['mir'].values))

    MIR_DICT = {}
    for row in MIRNAS.iterrows():
        guide_seq = row[1]['guide_seq']
        pass_seq = row[1]['pass_seq']
        MIR_DICT[row[1]['mir']] = {
            'mirseq': guide_seq,
            'site8': utils.rev_comp(guide_seq[1:8]) + 'A',
            'one_hot': utils.one_hot_encode(guide_seq[:options.MIRLEN])
        }

        MIR_DICT[row[1]['mir'] + '*'] = {
            'mirseq': pass_seq,
            'site8': utils.rev_comp(pass_seq[1:8]) + 'A',
            'one_hot': utils.one_hot_encode(pass_seq[:options.MIRLEN])
        }

    ### READ EXPRESSION DATA ###
    TPM = pd.read_csv(options.TPM_FILE, sep='\t', index_col=0).sort_index()
    for mir in ALL_GUIDES:
        if mir not in TPM.columns:
            raise ValueError('{} given in mirseqs file but not in TPM file.'.format(mir))

    num_batches = 10
    TPM['batch'] = [ix % num_batches for ix in TPM['ix']]

    print("Using mirs: {}".format(ALL_GUIDES))

    # read in orf sequences
    ORF_SEQS = pd.read_csv(options.ORF_FILE, sep='\t', header=None, index_col=0)

    feature_names = ['mir', 'tpm', 'orf_guide_1hot', 'utr3_guide_1hot', 
                     'orf_pass_1hot', 'utr3_pass_1hot']

    with tf.python_io.TFRecordWriter(options.OUTFILE) as tfwriter:
        for ix, row in enumerate(TPM.iterrows()):

            # print progress
            if ix % 100 == 0:
                print("Processed {}/{} transcripts".format(ix, len(TPM)))

            transcript = row[0]
            utr3 = row[1]['sequence']
            orf = ORF_SEQS.loc[transcript][2]
            transcript_sequence = orf + utr3
            orf_length = len(orf)

            context_dict = tf.train.Features(feature={
                'transcript': tf_utils._bytes_feature(transcript.encode('utf-8')),
                'batch': tf_utils._int64_feature([row[1]['batch']])
            })

            total_transcript_sites = 0

            features = [[], [], [], [], [], []]
            for mir in ALL_GUIDES:

                site8 = MIR_DICT[mir]['site8']
                mirseq = MIR_DICT[mir]['mirseq']

                site8_star = MIR_DICT[mir + '*']['site8']
                mirseq_star = MIR_DICT[mir + '*']['mirseq']

                features[0].append(tf_utils._bytes_feature(mir.encode('utf-8')))  # mir
                features[1].append(tf_utils._float_feature([row[1][mir]]))  # tpm

                # get sites for guide strand
                seqs, locs = get_site_features.get_sites_from_utr(transcript_sequence, site8, overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)
                num_orf_sites = len([l for l in locs if l < orf_length])
                orf_sites = utils.mir_site_pair_to_ints(mirseq[:options.MIRLEN], ''.join(seqs[:num_orf_sites]))
                utr3_sites = utils.mir_site_pair_to_ints(mirseq[:options.MIRLEN], ''.join(seqs[num_orf_sites:]))
                features[2].append(tf_utils._int64_feature(orf_sites))
                features[3].append(tf_utils._int64_feature(utr3_sites))
                total_transcript_sites += len(locs)

                # get sites for guide strand
                seqs, locs = get_site_features.get_sites_from_utr(transcript_sequence, site8_star, overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)
                num_orf_sites = len([l for l in locs if l < orf_length])
                orf_sites = utils.mir_site_pair_to_ints(mirseq_star[:options.MIRLEN], ''.join(seqs[:num_orf_sites]))
                utr3_sites = utils.mir_site_pair_to_ints(mirseq_star[:options.MIRLEN], ''.join(seqs[num_orf_sites:]))
                features[4].append(tf_utils._int64_feature(orf_sites))
                features[5].append(tf_utils._int64_feature(utr3_sites))
                total_transcript_sites += len(locs)

                # features[0].append(tf_utils._bytes_feature(mir.encode('utf-8')))  # mir
                # features[1].append(tf_utils._float_feature([row[1][mir]]))  # tpm
                # features[2].append(tf_utils._int64_feature(utils.one_hot_encode(mirseq[:options.MIRLEN])))  # mirseq
                # assert len(utils.one_hot_encode(mirseq[:options.MIRLEN])) == 40

                # # get sites for guide strand
                # seqs, locs = get_site_features.get_sites_from_utr(transcript_sequence, site8, overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)
                # num_orf_sites = len([l for l in locs if l < orf_length])
                # orf_sites = ''.join(seqs[:num_orf_sites])
                # utr3_sites = ''.join(seqs[num_orf_sites:])
                # features[3].append(tf_utils._int64_feature(utils.one_hot_encode(orf_sites)))
                # features[4].append(tf_utils._int64_feature(utils.one_hot_encode(orf_sites)))
                # total_transcript_sites += len(locs)

                # features[5].append(tf_utils._int64_feature(utils.one_hot_encode(mirseq_star[:options.MIRLEN])))  # mirseq*
                # assert len(utils.one_hot_encode(mirseq_star[:options.MIRLEN])) == 40

                # # get sites for guide strand
                # seqs, locs = get_site_features.get_sites_from_utr(transcript_sequence, site8_star, overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)
                # num_orf_sites = len([l for l in locs if l < orf_length])
                # orf_sites = ''.join(seqs[:num_orf_sites])
                # utr3_sites = ''.join(seqs[num_orf_sites:])
                # features[6].append(tf_utils._int64_feature(utils.one_hot_encode(orf_sites)))
                # features[7].append(tf_utils._int64_feature(utils.one_hot_encode(orf_sites)))
                # total_transcript_sites += len(locs)

            print(total_transcript_sites)

            if total_transcript_sites > 0:

                feature_dict = tf.train.FeatureLists(feature_list={
                    feature_names[ix]: tf.train.FeatureList(feature=features[ix]) for ix in range(len(feature_names))
                })

                # Create the SequenceExample
                example = tf.train.SequenceExample(context=context_dict,
                                                   feature_lists=feature_dict)

                tfwriter.write(example.SerializeToString())
            else:
                print('Skipping {} because no sites found'.format(transcript))
