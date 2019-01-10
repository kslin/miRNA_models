from optparse import OptionParser
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import config, helpers

np.set_printoptions(threshold=np.inf, linewidth=200)
pd.options.mode.chained_assignment = None


def encode_duplex(mirseq, sites):
    """
    Encodes sequences into feature matrix
    """

    rev_mirseq = mirseq[:config.MIRLEN][::-1]
    mirseq_one_hot = helpers.one_hot_encode(rev_mirseq, config.MIR_NT_DICT, config.TARGETS)

    duplex_one_hot = np.zeros([len(sites), 4*config.MIRLEN, 4*config.SEQLEN])
    for ix, siteseq in enumerate(sites):
        site_one_hot = helpers.one_hot_encode(siteseq, config.SEQ_NT_DICT, config.TARGETS)
        duplex_one_hot[ix,:] = np.outer(mirseq_one_hot, site_one_hot)
    
    return duplex_one_hot


def get_ts7_features(mirseq, locs, utr, utr_len, orf_len, upstream_limit, rnaplfold_data):
    # calculate TS7 features
    features = []
    for loc in locs:

        # get ts7 features
        local_au = helpers.calculate_local_au(utr, loc-3)
        threep = helpers.calculate_threep_score(mirseq, utr, loc-3, upstream_limit)
        min_dist = min(loc, utr_len - (loc + 6))
        assert (min_dist >= 0), (loc, utr_len)

        # use the rnaplfold data to calculate the site accessibility
        site_start_for_SA = loc + 7
        if (site_start_for_SA) not in rnaplfold_data.index:
            sa_score = 0
        else:
            row_vals = rnaplfold_data.loc[site_start_for_SA].values[:14] # pos 1-14 unpaired, Agarwal 2015
            # row_vals = rnaplfold_data.loc[site_start_for_SA].values[:10] # pos 1-10 unpaired, Sean

            for raw_sa_score in row_vals[::-1]:
                if not np.isnan(raw_sa_score):
                    break

            if np.isnan(raw_sa_score):
                sa_score = np.nan
            elif raw_sa_score <= 0:
                sa_score = -5.0
                print("warning, nan sa_score")
            else:
                sa_score = np.log10(raw_sa_score)

        features.append([local_au, threep, min_dist, sa_score, utr_len, orf_len])

    return features


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-m", "--merged", dest="MERGED", help="merged file")
    parser.add_option("-o", "--orffile", dest="ORF_FILE", help="ORF sequences in tsv format")
    parser.add_option("-t", "--test_mirna", dest="TEST_MIRNA", help="test miRNA name")
    parser.add_option("-r", "--rnaplfold_folder", dest="RNAPLFOLD_FOLDER", help="location of RNAPLfold lunp files")
    parser.add_option("-w", "--outfile", dest="OUTFILE", help="location for tfrecords")
    parser.add_option("--overlap_dist", dest="OVERLAP_DIST", help="minimum distance between neighboring sites", type=int)
    parser.add_option("--upstream_limit", dest="UPSTREAM_LIMIT", help="how far upstream to look for 3p pairing", type=int)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--only_canon", dest="ONLY_CANON", help="only use canonical sites", default=False, action='store_true')

    (options, args) = parser.parse_args()

    # get train miRNAs
    if options.TEST_MIRNA == 'none':
        train_mirs = config.MIRS16
    else:
        train_mirs = [m for m in config.MIRS16 if m != options.TEST_MIRNA]

    print('Number of training miRNAs: {}'.format(len(train_mirs)))

    # read in orf tpms and utr sequences
    MERGED = pd.read_csv(options.MERGED, sep='\t', index_col=0)

    # read in orf sequences
    ORF_SEQS = pd.read_csv(options.ORF_FILE, sep='\t', header=None, index_col=0)

    with tf.python_io.TFRecordWriter(options.OUTFILE) as tfwriter:
        for ix, row in enumerate(MERGED.iterrows()):

            # print progress
            if ix % 100 == 0:
                print("processing {}/{} transcripts".format(ix, len(MERGED)))

            transcript = row[0]
            utr = row[1]['sequence']
            utr_ext = ('TTT' + utr + 'TTT')
            orf = ORF_SEQS.loc[transcript][2]

            lunp_file = os.path.join(options.RNAPLFOLD_FOLDER, transcript) + '_lunp'
            rnaplfold_data = pd.read_csv(lunp_file, sep='\t', header=1).set_index(' #i$').astype(float)
            
            utr_len = len(utr)
            orf_len = len(orf)

            duplex_matrix = []
            feature_matrix = []
            nsites = []

            for mir in train_mirs:

                sitem8 = config.SITE_DICT[mir]
                mirseq = config.MIRSEQ_DICT[mir]

                # get sites and 12mer sequences
                locs = helpers.get_locs(utr, sitem8, overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)
                nsites.append(len(locs))

                if len(locs) > 0:
                    seqs = [utr_ext[l:l+config.SEQLEN] for l in locs]
                    duplex_matrix.append(encode_duplex(mirseq, seqs))

                    features = get_ts7_features(mirseq, locs, utr, utr_len, orf_len, options.UPSTREAM_LIMIT, rnaplfold_data)
                    feature_matrix += features

                # if indicated, add in features for passenger strand too
                if options.PASSENGER:
                    sitem8 = config.SITE_DICT[mir + '*']
                    mirseq = config.MIRSEQ_DICT[mir + '*']

                    # get sites and 12mer sequences
                    locs = helpers.get_locs(utr, sitem8, overlap_dist=options.OVERLAP_DIST, only_canon=options.ONLY_CANON)
                    nsites.append(len(locs))

                    if len(locs) > 0:
                        seqs = [utr_ext[l:l+config.SEQLEN] for l in locs]
                        duplex_matrix.append(encode_duplex(mirseq, seqs))

                        features = get_ts7_features(mirseq, locs, utr, utr_len, orf_len, options.UPSTREAM_LIMIT, rnaplfold_data)
                        feature_matrix += features

                # break

            if np.sum(nsites) == 0:
                continue

            # concat duplex features and center
            duplex_matrix = (np.concatenate(duplex_matrix, axis=0) * 4) - 0.25 # nsites x 4mirlen x 4seqlen
            feature_matrix = np.array(feature_matrix) # nsites x num_features
            nsites = np.array(nsites)
            tpms = row[1][train_mirs].values

            assert (duplex_matrix.shape[0] == np.sum(nsites))
            assert (feature_matrix.shape[0] == np.sum(nsites))


            feature_dict = {
                'transcript': helpers._bytes_feature(transcript.encode('utf-8')),
                'duplex_features': helpers._float_feature(duplex_matrix.flatten()),
                'ts7_features': helpers._float_feature(feature_matrix.flatten()),
                'nsites': helpers._int64_feature(nsites),
                'tpms': helpers._float_feature(tpms),
            }

            # Create a Features message using tf.train.Example.
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            example_proto = example_proto.SerializeToString()

            tfwriter.write(example_proto)

            # break



