from optparse import OptionParser

import numpy as np
import pandas as pd
import tensorflow as tf

import utils
import tf_utils


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-w", "--outfile", dest="OUTFILE", help="location for tfrecords")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--seqlen", dest="SEQLEN", type=int)
    parser.add_option("--num_examples", dest="NUM_EXAMPLES", type=int)

    (options, args) = parser.parse_args()

    log_kd_dict = {
        '8mer': -6.0,
        '7mer-m8': -5.0,
        '7mer-a1': -4.0,
        '6mer': -2.5,
        '6mer-m8': -2.0,
        '6mer-a1': -1.5,
        'no site': 0.0,
    }

    mir = "random".encode('utf-8')
    with tf.python_io.TFRecordWriter(options.OUTFILE) as tfwriter:
        for _ in range(options.NUM_EXAMPLES):

                mirseq = utils.generate_random_seq(options.MIRLEN)
                site8 = utils.rev_comp(mirseq[1:8]) + 'A'

                if np.random.random() > 0.5:
                    while True:
                        siteseq = utils.generate_random_seq(4) + site8[2:-2] + utils.generate_random_seq(4)
                        stype = utils.get_centered_stype(site8, siteseq)
                        if stype != 'no site':
                            break

                else:
                    siteseq = utils.generate_random_seq(options.SEQLEN)
                    stype = utils.get_centered_stype(site8, siteseq)

                log_kd = log_kd_dict[stype] + np.random.normal(scale=0.2) + 0.2 + 0.175
                log_kd += (siteseq[:2] + siteseq[-2:]).count('A') * (-0.1)
                log_kd += (siteseq[:2] + siteseq[-2:]).count('T') * (-0.1)
                log_kd += (siteseq[2:-3]).count('C') * (-0.05)
                log_kd += (siteseq[2:-3]).count('G') * (-0.05)

                feature_dict = {
                    'mir': tf_utils._bytes_feature(mir),
                    'mir_1hot': tf_utils._float_feature(utils.one_hot_encode(mirseq)),
                    'seq_1hot': tf_utils._float_feature(utils.one_hot_encode(siteseq)),
                    'log_kd': tf_utils._float_feature([log_kd]),
                    'keep_prob': tf_utils._float_feature([1.0]),
                    'stype': tf_utils._bytes_feature(stype.encode('utf-8')),
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                example_proto = example_proto.SerializeToString()

                tfwriter.write(example_proto)
