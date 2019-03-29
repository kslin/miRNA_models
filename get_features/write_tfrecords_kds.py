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

    print("Length of KD data before removing sites in other registers: {}".format(len(KDS)))
    KDS = KDS[KDS['best_stype'] == KDS['aligned_stype']]
    print("Length of KD data after removing sites in other registers: {}".format(len(KDS)))

    # balance data
    KDS['nearest'] = np.minimum(0, np.round(KDS['log_kd'] * 4) / 4)
    KDS['count'] = 1
    temp = KDS.groupby('nearest').agg({'count': np.sum})
    temp['target'] = np.exp(temp.index + 5) * 500
    temp['keep_prob'] = np.minimum(1.0, temp['target'] / temp['count'])
    temp['keep_prob'] = [1.0 if x < -3 else y for (x, y) in zip(temp.index, temp['keep_prob'])]
    temp_dict = {x: y for (x, y) in zip(temp.index, temp['keep_prob'])}
    KDS['keep_prob'] = [temp_dict[x] for x in KDS['nearest']]

    KDS = KDS.drop(['nearest', 'count'], 1)

    if options.ONLY_CANON:
        KDS = KDS[KDS['aligned_stype'] != 'no site']

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
                keep_prob = row[1]['keep_prob']
                stype = row[1]['aligned_stype']

                feature_dict = {
                    'mir': utils._bytes_feature(mir.encode('utf-8')),
                    'mir_1hot': utils._float_feature(utils.one_hot_encode(mirseq[:options.MIRLEN])),
                    'seq_1hot': utils._float_feature(utils.one_hot_encode(siteseq)),
                    'log_kd': utils._float_feature([log_kd]),
                    'keep_prob': utils._float_feature([keep_prob]),
                    'stype': utils._bytes_feature(stype.encode('utf-8')),
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                example_proto = example_proto.SerializeToString()

                tfwriter.write(example_proto)
