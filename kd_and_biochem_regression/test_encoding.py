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
NUM_TRAIN = len(TRAIN_GUIDES)

print(ALL_GUIDES)
print(TRAIN_GUIDES)

if PASSENGER:
    TRAIN_MIRS = np.array(list(zip(TRAIN_GUIDES, [x+'*' for x in TRAIN_GUIDES]))).flatten().tolist()
else:
    TRAIN_MIRS = TRAIN_GUIDES

ALL_MIRS = np.array(list(zip(ALL_GUIDES, [x+'*' for x in ALL_GUIDES]))).flatten().tolist()

print(TRAIN_MIRS)


init_params = [
    -4.0,  # FREEAGO_INIT,
    0.0,   # GUIDE_OFFSET_INIT,
    -1.0,  # PASS_OFFSET_INIT,
    -0.5,  # DECAY_INIT,
    -8.5,  # UTR_COEF_INIT
]

_freeAGO_mean = tf.get_variable('freeAGO_mean', shape=(), initializer=tf.constant_initializer(init_params[0]))
_freeAGO_guide_offset = tf.get_variable('freeAGO_guide_offset', shape=[NUM_TRAIN, 1],
                        initializer=tf.constant_initializer(init_params[1]))
_freeAGO_pass_offset = tf.get_variable('freeAGO_pass_offset', shape=[NUM_TRAIN, 1], initializer=tf.constant_initializer(init_params[2]))
_freeAGO_all = tf.reshape(tf.concat([_freeAGO_guide_offset + _freeAGO_mean, _freeAGO_pass_offset + _freeAGO_mean], axis=1), [NUM_TRAIN * 2], name='freeAGO_all')


filename = "/lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/guide_passenger_only_canon.tfrecord"
raw_dataset = tf.data.TFRecordDataset(filename)

parsed_dataset = raw_dataset.map(lambda x: parse_data._parse_repression_function(x, TRAIN_MIRS, ALL_MIRS, MIRLEN, SEQLEN, NUM_FEATS, _freeAGO_all))
iterator = parsed_dataset.make_initializable_iterator()
next_batch = iterator.get_next()


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(iterator.initializer)
#     print(sess.run(_freeAGO_all))
#     for _ in range(3):
#         x, y, z, a, b = sess.run(next_batch)
#         # print(a)
#         print(x.shape[0], y.shape[0], np.sum(a))

#     print(x.shape)
#     print(y.shape)
#     print(z.shape)
#     print(a)
#     print(b)

    # for _ in range(2):
    #     sess.run(iterator.initializer)
    #     while True:
    #         try:
    #             x, y, z, a = sess.run(next_batch)
    #         except tf.errors.OutOfRangeError:
    #             break

    #     print('finished epoch')

results = parse_data._build_tpm_batch(iterator, 5)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    results = sess.run(results)
    print(results['images'].shape)
    print(results['features'].shape)
    print(results['labels'].shape)
    print(np.sum(results['nsites']))
    print(results['nsites'])
    print(results['freeAGOs'])


# filename = '/lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/log_kds.tfrecord'
# parsed_dataset = tf.data.TFRecordDataset(filename)
# parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)
# parsed_dataset = parsed_dataset.map(parse_data._parse_log_kd_function)
# parsed_dataset = parsed_dataset.filter(lambda x, y, z: tf.math.logical_not(tf.equal(x, TEST_MIR.encode('utf-8'))))
# parsed_dataset = parsed_dataset.batch(20)
# iterator = parsed_dataset.make_initializable_iterator()
# next_batch_x, next_batch_y, next_batch_z = iterator.get_next()

# with tf.Session() as sess:
#     sess.run(iterator.initializer)
#     x, y, z = sess.run([next_batch_x, next_batch_y, next_batch_z])
#     print(x, z)
#     print(y.shape, z.shape)
#     # print(x.shape)
