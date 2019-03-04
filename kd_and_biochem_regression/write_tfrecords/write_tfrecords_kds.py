from optparse import OptionParser

import numpy as np
import pandas as pd
import tensorflow as tf

import utils


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-k", "--kdfile", dest="KD_FILE", help="kd data")
    parser.add_option("-w", "--outfile", dest="OUTFILE", help="location for tfrecords")
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--only_canon", dest="ONLY_CANON", help="only use canonical sites", default=False, action='store_true')

    (options, args) = parser.parse_args()

    KDS = pd.read_csv(options.KD_FILE, sep='\t')

    if options.ONLY_CANON:
        KDS = KDS[KDS['aligned_stype'] != 'no site']

    print("Length of KD data before removing sites in other registers: {}".format(len(KDS)))
    KDS = KDS[KDS['best_stype'] == KDS['aligned_stype']]
    print("Length of KD data after removing sites in other registers: {}".format(len(KDS)))

    print(KDS.head())

    for mir, group in KDS.groupby('mir'):
        shuffle_ixs = np.random.permutation(len(group))
        group = group.iloc[shuffle_ixs]
        print("Processing {}".format(mir))
        with tf.python_io.TFRecordWriter(options.OUTFILE + '_{}.tfrecord'.format(mir)) as tfwriter:
            for ix, row in enumerate(group.iterrows()):

                # print progress
                if ix % 10000 == 0:
                    print("Processed {}/{} KDS".format(ix, len(group)))

                mirseq = row[1]['mirseq']
                siteseq = row[1]['12mer']
                log_kd = row[1]['log_kd']

                feature_dict = {
                    'mir': utils._bytes_feature(mir.encode('utf-8')),
                    'mir_1hot': utils._float_feature(utils.one_hot_encode(mirseq[:options.MIRLEN])),
                    'seq_1hot': utils._float_feature(utils.one_hot_encode(siteseq)),
                    'log_kd': utils._float_feature([log_kd]),
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                example_proto = example_proto.SerializeToString()

                tfwriter.write(example_proto)
