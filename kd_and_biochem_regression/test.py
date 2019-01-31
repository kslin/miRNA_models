import numpy as np
import pandas as pd
import tensorflow as tf

import model
import parse_data_utils


### Test functions from model.py ###
def test_pad_utr_kds():
    batch_size = 2
    num_mirs = 6
    vals = tf.constant(np.array([1,4,6,6,3,2,5,4,2,3,3,4,4,1]), dtype=tf.float32)
    split_sizes = tf.constant(np.array([2,0,1,2,0,2,1,1,1,0,3,1]), dtype=tf.int32)

    results = model.pad_vals(vals, split_sizes, num_mirs, batch_size)

    with tf.Session() as sess:
        np_results = sess.run(results)

    expected = np.array(
        [[[   1.,    4., -100.],
          [-100., -100., -100.],
          [   6., -100., -100.],
          [   6.,    3., -100.],
          [-100., -100., -100.],
          [   2.,    5., -100.]],

         [[   4., -100., -100.],
          [   2., -100., -100.],
          [   3., -100., -100.],
          [-100., -100., -100.],
          [   3.,    4.,    4.],
          [   1., -100., -100.]]]
    )

    assert (np.sum(np.abs(np_results - expected)) < 0.0001)

# test_pad_utr_kds()


def test_get_pred_logfc():
    batch_size = 2
    num_guides = 3
    num_feats = 2
    passenger = True
    utr_ka_values = tf.constant(np.array([[1],[4],[6],[6],[3],[2],[5],[4],[2],[3],[3],[4],[4],[1]]), dtype=tf.float32)
    utr_split_sizes = tf.constant(np.array([2,0,1,2,0,2,1,1,1,0,3,1]), dtype=tf.int32)
    ts7_feats = tf.constant(np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]), dtype=tf.float32)
    freeAGO_all = tf.constant(np.array([-4,-5,-4,-5,-4,-5]), dtype=tf.float32)
    ts7_weights = tf.constant(np.array([[1],[2],[3]]), dtype=tf.float32)
    labels = tf.constant(np.array([[-1.8741043, -2.9491906, -1.8741043],
                                    [-1.8741043, -0.98201376, -3.6996868]]), dtype=tf.float32)

    tpm_batch = {'nsites': utr_split_sizes, 'features': ts7_feats, 'labels': labels}

    results1 = model.get_pred_logfc(utr_ka_values, freeAGO_all, tpm_batch, ts7_weights, batch_size, passenger, num_guides, 'test', 'MEAN_CENTER')
    results2 = model.get_pred_logfc_old(utr_ka_values, freeAGO_all, tpm_batch, ts7_weights, batch_size, passenger, num_guides, 'test', 'MEAN_CENTER')

    with tf.Session() as sess:
        np_results = sess.run(results2)

    assert (np.sum(np.abs(np_results[1] - np_results[2])) < 0.0001)
    assert (np.abs(np.mean(np_results[1])) < 0.0001)
    assert (np.abs(np.mean(np_results[2])) < 0.0001)

    # print(np_results)

test_get_pred_logfc()
