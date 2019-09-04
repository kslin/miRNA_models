import itertools as it
from optparse import OptionParser

import numpy as np
import pandas as pd
import tensorflow as tf

import utils
import tf_utils

def generate_12mers(site8):
    all_8mers = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"],repeat=8))]
    mers = []
    for i in range(5):
        subseq = site8[i:i+4]
        mers += [x[:i+2] + subseq + x[i+2:] for x in all_8mers]
    mers = list(set(mers))
    return sorted(mers)


def write_12mers(mirname, mirseq, outfile):

    site8 = utils.rev_comp(mirseq[1:8]) + 'A'
    all_12mers = generate_12mers(site8)

    if len(all_12mers) != 262144:
        raise(ValueError("all_12mers should be 262144 in length"))

    with tf.python_io.TFRecordWriter(outfile) as tfwriter:
        for siteseq in all_12mers:

            aligned_stype = utils.get_centered_stype(site8, siteseq)
            if aligned_stype == 'no site':
                keep_prob = 0.001
            else:
                keep_prob = 1.0

            feature_dict = {
                'mir': tf_utils._bytes_feature(mirname.encode('utf-8')),
                'mir_1hot': tf_utils._float_feature(utils.one_hot_encode(mirseq)),
                'seq_1hot': tf_utils._float_feature(utils.one_hot_encode(siteseq)),
                'log_kd': tf_utils._float_feature([-0.0]),
                'keep_prob': tf_utils._float_feature([keep_prob]),
                'stype': tf_utils._bytes_feature(aligned_stype.encode('utf-8')),
            }

            example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            example_proto = example_proto.SerializeToString()

            tfwriter.write(example_proto)


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--mirseqs", dest="MIRSEQS", help="miRNA dataframe")
    parser.add_option("--outfile", dest="OUTFILE", help="location for tfrecords")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)

    (options, args) = parser.parse_args()

    MIRNA_DATA = pd.read_csv(options.MIRSEQS, sep='\t', index_col='mir')

    for row in MIRNA_DATA.iterrows():
        MIRNAME = row[0]

        # if no rbns data, get 12mers for guide strand
        if not row[1]['has_rbns']:
            print(MIRNAME)
            MIRSEQ = row[1]['guide_seq'][:options.MIRLEN]
            write_12mers(MIRNAME, MIRSEQ, options.OUTFILE + '_{}.tfrecord'.format(MIRNAME))

        # for all miRNAs, get 12mers for passenger strand
        print(MIRNAME + '_pass')
        MIRSEQ = row[1]['pass_seq'][:options.MIRLEN]
        write_12mers(MIRNAME + '_pass', MIRSEQ, options.OUTFILE + '_{}_pass.tfrecord'.format(MIRNAME))
