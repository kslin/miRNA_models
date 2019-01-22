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


def calculate_threep_score(mirseq, utr, site_start, upstream_limit):
    """
    Calculate the three-prime pairing score

    Parameters
    ----------
    mirseq: string, miRNA sequence
    utr: string, utr sequence
    site_start: int, start of 12mer site
    upstream_limit: int, how far upstream to look for 3p pairing

    Output
    ------
    float: 3' pairing score
    """
    if site_start <= 0:
        return 0

    # get the 3' region of the mirna and the corresponding utr seq
    mirseq_3p = mirseq[8:]  # miRNA sequence from position 9 onward
    trailing = utr[max(0, site_start - upstream_limit): site_start + 2]  # site sequence up to edges of possible 8mer site
    utr_5p = utils.rev_comp(trailing)

    # initiate array for dynamic programming search
    scores = np.empty((len(utr_5p) + 1, len(mirseq_3p) + 1))
    scores.fill(np.nan)
    possible_scores = [0]

    # fill in array
    for i, nt1 in enumerate(utr_5p):
        for j, nt2 in enumerate(mirseq_3p):
            if nt1 == nt2:
                new_score = 0.5 + 0.5 * ((j > 3) & (j < 8))
                if not np.isnan(scores[i, j]):
                    new_score += scores[i, j]
                    scores[i + 1, j + 1] = new_score
                    possible_scores.append(new_score)
                else:
                    offset_penalty = max(0, (abs(i - j) - 2) * 0.5)
                    scores[i + 1, j + 1] = new_score - offset_penalty
            else:
                scores[i + 1, j + 1] = float('NaN')

    return np.nanmax(possible_scores)


def calculate_local_au(utr, site_start):
    """
    Calculate the local AU score

    Parameters
    ----------
    utr: string, utr sequence
    site_start: int, start of 12mer site

    Output
    ------
    float: local AU score
    """
    # find A, U and weights upstream of site
    upstream = utr[max(0, site_start - 30): max(0, site_start)]
    upstream = [int(x in ['A', 'U']) for x in upstream]
    upweights = [1.0 / (x + 1) for x in range(len(upstream))][::-1]

    # find A,U and weights downstream of site
    downstream = utr[site_start + 12:min(len(utr), site_start + 42)]
    downstream = [int(x in ['A', 'U']) for x in downstream]
    downweights = [1.0 / (x + 1) for x in range(len(downstream))]

    weighted = np.dot(upstream, upweights) + np.dot(downstream, downweights)
    total = float(sum(upweights) + sum(downweights))

    return weighted / total


def get_ts7_features(mirseq, locs, utr, utr_len, orf_len, upstream_limit, rnaplfold_data):
    # calculate TS7 features
    features = []
    for loc in locs:

        # get ts7 features
        local_au = calculate_local_au(utr, loc - 3)
        threep = calculate_threep_score(mirseq, utr, loc - 3, upstream_limit)
        min_dist = min(loc, utr_len - (loc + 6))
        assert (min_dist >= 0), (loc, utr_len)

        # use the rnaplfold data to calculate the site accessibility
        site_start_for_SA = loc + 7
        if (site_start_for_SA) not in rnaplfold_data.index:
            sa_score = 0
        else:
            row_vals = rnaplfold_data.loc[site_start_for_SA].values[:14]  # pos 1-14 unpaired, Agarwal 2015
            # row_vals = rnaplfold_data.loc[site_start_for_SA].values[:10]  # pos 1-10 unpaired, Sean

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

        features += [local_au, threep, min_dist, sa_score, utr_len, orf_len]

    return np.array(features).astype(float)


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-t", "--tpmfile", dest="TPM_FILE", help="tpm data")
    parser.add_option("-m", "--mirseqs", dest="MIR_SEQS", help="tsv with miRNAs and their sequences")
    # parser.add_option("-o", "--orffile", dest="ORF_FILE", help="ORF sequences in tsv format")
    parser.add_option("-r", "--rnaplfold_folder", dest="RNAPLFOLD_FOLDER", help="location of RNAPLfold lunp files")
    parser.add_option("-w", "--outfile", dest="OUTFILE", help="location for tfrecords")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--overlap_dist", dest="OVERLAP_DIST", help="minimum distance between neighboring sites", type=int)
    parser.add_option("--upstream_limit", dest="UPSTREAM_LIMIT", help="how far upstream to look for 3p pairing", type=int)
    parser.add_option("--passenger", dest="PASSENGER", help="include passenger", default=False, action='store_true')
    parser.add_option("--only_canon", dest="ONLY_CANON", help="only use canonical sites", default=False, action='store_true')
    parser.add_option("--write_seqs", dest="WRITE_SEQS", help="write seqs found", default=None)

    (options, args) = parser.parse_args()

    ### READ miRNA DATA ###
    MIRNAS = pd.read_csv(options.MIR_SEQS, sep='\t')
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

    # read in orf sequences
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
            utr_ext = ('TTT' + utr + 'TTT')

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

                    features = get_ts7_features(mirseq, locs, utr, utr_len, orf_len, options.UPSTREAM_LIMIT, rnaplfold_data)
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

