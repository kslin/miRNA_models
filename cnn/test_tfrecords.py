import numpy as np
# import pandas as pd
import tensorflow as tf

import parse_data_utils, model

# tf.enable_eager_execution()


tfrecord_file = '../data/cnn/tfrecords/guide_passenger_withORFcanon_allsites_feat8_11batches_SAbg.tfrecord'
tpm_dataset = tf.data.TFRecordDataset(tfrecord_file)

TRAIN_MIRS = ['lsy6','mir1']
VAL_MIRS = ['mir7']

MIRLEN, SEQLEN = 10, 12


def _parse_fn_train(x):
        return parse_data_utils._parse_repression_function(x, TRAIN_MIRS, MIRLEN, SEQLEN, 8, 2, True)

def _parse_fn_val(x):
    return parse_data_utils._parse_repression_function(x, TRAIN_MIRS + VAL_MIRS, MIRLEN, SEQLEN, 8, 2, True)

tpm_dataset = tpm_dataset.map(_parse_fn_train)
tpm_iterator = tpm_dataset.make_initializable_iterator()
next_tpm_batch = parse_data_utils._build_tpm_batch(tpm_iterator, 1)

padded = model.pad_vals(tf.squeeze(next_tpm_batch['features'][:, 1]), next_tpm_batch['nsites'], 4, 1, -50)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tpm_iterator.initializer)

    print(sess.run(next_tpm_batch))
