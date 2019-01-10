from optparse import OptionParser
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import config, helpers, data_objects, tf_helpers

pd.options.mode.chained_assignment = None


def serialize_kd_data(mirseq, siteseq, logkd):
    """
    Creates a tf.Example message ready to be written to a file for KD data.
    """

    rev_mirseq = mirseq[:config.MIRLEN][::-1]

    mirseq_one_hot = helpers.one_hot_encode(rev_mirseq, config.MIR_NT_DICT, config.TARGETS)
    siteseq_one_hot = helpers.one_hot_encode(siteseq, config.SEQ_NT_DICT, config.TARGETS)

    feature = {
      'mirseq': _float_feature(mirseq_one_hot),
      'siteseq': _float_feature(siteseq_one_hot),
      'kd': _float_feature(logkd),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-k", "--kd_file", dest="KD_FILE", help="KD data in tsv format")
    parser.add_option("-c", "--kd_cutoff", dest="KD_CUTOFF", help="which KD")
    parser.add_option("-m", "--mirname", dest="MIRNAME", help="miRNA name")
    parser.add_option("-o", "--outfile", dest="OUTFILE", help="location for tfrecords")
    parser.add_option("--mode", dest="MODE", help="KD or UTR")

    (options, args) = parser.parse_args()

    with open(options.KD_FILE, 'r') as infile:
        for line in infile:
            transcript, utr = line.strip('\n').split('\t')







