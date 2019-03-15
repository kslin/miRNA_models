import numpy as np
import pandas as pd
import tensorflow as tf

import model
import parse_data_utils
import utils


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

def np_pad(split_sizes, values, final_shape):
    results = np.zeros(final_shape) - 100.0
    ix = 0
    for j, size in enumerate(split_sizes):
        results[j, :size] = values[ix: ix+size]
        ix += size

    return results


def test_get_pred_logfc():
    batch_size = 2
    num_guides = 3
    num_feats = 2
    passenger = True

    # len(utr_split_sizes) = batch_size*num_guides*(2 if passenger)
    utr_split_sizes = np.array([2,0,1,2,0,2,1,1,1,0,3,1])

    # len(utr_ka_values) = nsites = sum(utr_split_sizes)
    utr_ka_values = np.array([[1],[4],[6],[6],[3],[2],[5],[4],[2],[3],[3],[4],[4],[1]], dtype=float)
    utr_ka_values_padded = np.array(
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
    
    # shape(ts7_feats) = [nsites, num_feats]
    np.random.seed(0)
    ts7_feats = np.random.choice([-0.5, 0.0, 0.5], size=num_feats*np.sum(utr_split_sizes)).reshape([-1, 2])
    freeAGO_all = np.array([-4,-5,-4,-5,-4,-5], dtype=float)
    ts7_weights = np.array([[1],[2]], dtype=float)
    ts7_bias = 1.0
    decay = -1.0

    occupancy = 1 / (1 + np.exp(-1 * (utr_ka_values_padded + freeAGO_all.reshape([1,-1,1]))))
    ts7_term = (ts7_feats @ ts7_weights).flatten() + ts7_bias
    ts7_term_padded = np.maximum(0, np_pad(utr_split_sizes, ts7_term, [batch_size*num_guides*2, 3]).reshape([batch_size, num_guides*2, 3]))
    labels = np.sum(np.sum(decay * occupancy * ts7_term_padded, axis=2).reshape([batch_size, num_guides, 2]), axis=2)

    _utr_split_sizes = tf.constant(utr_split_sizes, dtype=tf.int32)
    _utr_ka_values = tf.constant(utr_ka_values, dtype=tf.float32)
    _ts7_feats = tf.constant(ts7_feats, dtype=tf.float32)
    _freeAGO_all = tf.constant(freeAGO_all, dtype=tf.float32)
    _ts7_weights = tf.constant(ts7_weights, dtype=tf.float32)
    _ts7_bias = tf.constant(ts7_bias, dtype=tf.float32)
    _decay = tf.constant(decay, dtype=tf.float32)
    _labels = tf.constant(labels, dtype=tf.float32)

    _tpm_batch = {
        'nsites': _utr_split_sizes,
        'features': _ts7_feats,
        'labels': _labels
    }

    results = model.get_pred_logfc_separate(_utr_ka_values, _freeAGO_all, _tpm_batch, _ts7_weights, _ts7_bias, _decay, batch_size, passenger, num_guides, 'test', 'MEAN_CENTER')

    with tf.Session() as sess:
        results_eval = sess.run(results)

    print(np.linalg.norm(results_eval[0] - labels))


test_get_pred_logfc()
