import sys

import numpy as np
import pandas as pd
import tensorflow as tf

import parse_data

np.set_printoptions(threshold=np.inf, linewidth=200)


PASSENGER = True
MIRNAS = pd.read_csv('write_tfrecords/mirseqs.txt', sep='\t')
ALL_GUIDES = list(MIRNAS['mir'].values)
TEST_MIR = 'mir124'
TRAIN_GUIDES = [m for m in ALL_GUIDES if m != TEST_MIR]
MIRLEN, SEQLEN = 10, 12
NUM_FEATS = 6

print(ALL_GUIDES)
print(TRAIN_GUIDES)

if PASSENGER:
    TRAIN_MIRS = np.array(list(zip(TRAIN_GUIDES, [x+'*' for x in TRAIN_GUIDES]))).flatten().tolist()
else:
    TRAIN_MIRS = TRAIN_GUIDES

ALL_MIRS = np.array(list(zip(ALL_GUIDES, [x+'*' for x in ALL_GUIDES]))).flatten().tolist()

print(TRAIN_MIRS)

filename = "/lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/guide_passenger_only_canon.tfrecord"
raw_dataset = tf.data.TFRecordDataset(filename)

parsed_dataset = raw_dataset.map(lambda x: parse_data._parse_repression_function(x, TRAIN_MIRS, ALL_MIRS, MIRLEN, SEQLEN, NUM_FEATS))
iterator = parsed_dataset.make_initializable_iterator()
next_batch = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    for _ in range(5):
        x, y, z, a = sess.run(next_batch)
        print(x.shape[0], y.shape[0], np.sum(a))

    print(x.shape)
    print(y.shape)
    print(z.shape)
    print(a.shape)

    # for _ in range(2):
    #     sess.run(iterator.initializer)
    #     while True:
    #         try:
    #             x, y, z, a = sess.run(next_batch)
    #         except tf.errors.OutOfRangeError:
    #             break

    #     print('finished epoch')

xs, ys, zs, ass = [], [], [], []
for _ in range(5):
    x, y, z, a = iterator.get_next()
    xs.append(x)
    ys.append(y)
    zs.append(z)
    ass.append(a)

xs = tf.concat(xs, axis=0)
ys = tf.concat(ys, axis=0)
zs = tf.concat(zs, axis=0)
ass = tf.concat(ass, axis=0)

with tf.Session() as sess:
    sess.run(iterator.initializer)
    xs, ys, zs, ass = sess.run([xs, ys, zs, ass])
    print(xs.shape)
    print(ys.shape)
    print(zs.shape)
    print(ass.shape)

# filename = '/lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/log_kds.tfrecord'
# raw_dataset = tf.data.TFRecordDataset(filename)
# parsed_dataset = raw_dataset.map(parse_data._parse_log_kd_function)
# parsed_dataset = parsed_dataset.batch(2)
# iterator = parsed_dataset.make_initializable_iterator()
# next_batch_x, next_batch_y = iterator.get_next()

# with tf.Session() as sess:
#     sess.run(iterator.initializer)
#     x, y = sess.run([next_batch_x, next_batch_y])
#     print(x.astype(int))
#     print(y)
#     print(x.shape)
