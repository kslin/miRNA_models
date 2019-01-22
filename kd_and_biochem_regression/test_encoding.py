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

blah1 = tf.convert_to_tensor(b'hi')
blah2 = tf.convert_to_tensor(b'hi')
hi = 'hi'

blah3 = tf.equal(blah1, hi.encode('utf-8'))

filename = '/lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/log_kds.tfrecord'
parsed_dataset = tf.data.TFRecordDataset(filename)
parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)
parsed_dataset = parsed_dataset.map(parse_data._parse_log_kd_function)
parsed_dataset = parsed_dataset.filter(lambda x, y, z: tf.math.logical_not(tf.equal(x, TEST_MIR.encode('utf-8'))))
parsed_dataset = parsed_dataset.batch(10)
iterator = parsed_dataset.make_initializable_iterator()
next_batch_x, next_batch_y, next_batch_z = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    x, y, z = sess.run([next_batch_x, next_batch_y, next_batch_z])
    print(x, z)
    print(sess.run(blah3))
    # print(x.shape)
