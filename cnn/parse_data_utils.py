import numpy as np
import pandas as pd
import tensorflow as tf


def outer_product_fn(mir_vector, seq_vector):
    dense_seq_vector = tf.sparse_tensor_to_dense(seq_vector)
    seq_vector_1hot = tf.reshape(tf.one_hot(dense_seq_vector, 4), [-1])
    dense_mir = tf.sparse_tensor_to_dense(mir_1hot)
    dense_seq = tf.sparse_tensor_to_dense(seq_1hot)
    image = tf.tensordot(dense_mir, dense_seq, axes=0)
    image = tf.transpose(tf.reshape(image, [mirlen, -1, seqlen]), perm=[1, 0, 2])
    return tf.contrib.layers.dense_to_sparse(image)


def _parse_repression_function(serialized_example, parse_guides, mirlen, seqlen, num_feats, use_feats, passenger):
    """Parse the serialized example from tfrecords
    Inputs:
        serialized_example: input of reading tfrecords
        parse_mirs (list of strings): list of miRNAs to parse, need to explicity include * if parsing star strands
        parse_mirs (list of strings): list of all miRNAs used to generate tfrecords in the correct order
        mirlen (int): length of miRNA sequence encoded
        seqlen (int): length of site sequence encoded
        num_feats (int): number of additional features
    """

    # make masks for extracting only data from parse_mirs
    if passenger:
        parse_mirs = []
        for guide in parse_guides:
            parse_mirs += [guide, guide + '*']
    else:
        parse_mirs = parse_guides

    # construct feature descriptions
    feature_description = {
        'transcript': tf.FixedLenFeature([], tf.string, default_value=''),
        'batch': tf.FixedLenFeature([], tf.int64, default_value=0),
        'utr3_length': tf.FixedLenFeature([], tf.float32, default_value=0),
        'orf_length': tf.FixedLenFeature([], tf.float32, default_value=0),
    }
    for guide in parse_guides:
        feature_description['{}_tpm'.format(guide)] = tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)

    for mir in parse_mirs:
        feature_description['{}_nsites'.format(mir)] = tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)
        feature_description['{}_mir_1hot'.format(mir)] = tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)
        feature_description['{}_seqs_1hot'.format(mir)] = tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)
        feature_description['{}_ts7_features'.format(mir)] = tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)


    parsed = tf.parse_single_example(serialized_example, feature_description)
    tpms = tf.concat([parsed['{}_tpm'.format(guide)] for guide in parse_guides], axis=0)
    nsites = tf.concat([parsed['{}_nsites'.format(mir)] for mir in parse_mirs], axis=0)
    images = []
    ts7_features = []
    for ix, mir in enumerate(parse_mirs):

        # perform outer product between miRNA 1hot and site 1hot, reshape to [num_sites, 4 * mirlen, 4 * seqlen]
        image = tf.tensordot(parsed['{}_mir_1hot'.format(mir)], parsed['{}_seqs_1hot'.format(mir)], axes=0)
        image = tf.transpose(tf.reshape(image, [4 * mirlen, -1, 4 * seqlen]), perm=[1, 0, 2])
        images.append(image)

        # reshape features to [num_sites, num_ts7_features]
        ts7_features.append(tf.reshape(parsed['{}_ts7_features'.format(mir)], [-1, num_feats])[:, :use_feats])

    images = tf.concat(images, axis=0)
    ts7_features = tf.concat(ts7_features, axis=0)

    # return parsed
    return images, ts7_features, tpms, tf.cast(nsites, tf.int32), parsed['transcript'], parsed['batch'], parsed['utr3_length'], parsed['orf_length']


def _build_tpm_batch(iterator, batch_size):
    images, features, labels, nsites, transcripts, batches, utr3_lengths, orf_lengths = [], [], [], [], [], [], [], []
    for ix in range(batch_size):
        results = iterator.get_next()

        images.append(results[0])
        features.append(results[1])
        labels.append(results[2])
        nsites.append(results[3])
        transcripts.append(results[4])
        batches.append(results[5])
        utr3_lengths.append(results[6])
        orf_lengths.append(results[7])

    results = {
        'images': tf.concat(images, axis=0),
        'features': tf.concat(features, axis=0),
        'labels': tf.stack(labels, axis=0),
        'nsites': tf.concat(nsites, axis=0),
        'transcripts': tf.stack(transcripts),
        'batches': tf.stack(batches),
        'utr3_length': tf.stack(utr3_lengths) / 5000,
        'orf_length': tf.stack(orf_lengths) / 5000,
    }

    return results


def _load_multiple_tfrecords(filenames):
    files = tf.data.Dataset.from_tensor_slices(tf.constant(filenames)).shuffle(buffer_size=10)
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(
        lambda filename: tf.data.TFRecordDataset(filename),
        cycle_length=len(filenames), sloppy=True)
    )
    return dataset


def _parse_log_kd_function(serialized_example):
    """Parse the serialized example read from tfrecords for KD data
    Inputs:
        serialized_example: input of reading tfrecords
    """

    # construct feature descriptions
    feature_description = {
        'mir': tf.FixedLenFeature([], tf.string, default_value=''),
        'mir_1hot': tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'seq_1hot': tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'log_kd': tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'keep_prob': tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'stype': tf.FixedLenFeature([], tf.string, default_value=''),
    }

    parsed = tf.parse_single_example(serialized_example, feature_description)
    image = tf.tensordot(parsed['mir_1hot'], parsed['seq_1hot'], axes=0)

    # return parsed data
    return parsed['mir'], image, parsed['log_kd'], parsed['keep_prob'], parsed['stype']


def _filter_kds_keepprob(mir, image, kd, keep_prob, stype):
    # return tf.math.greater(keep_prob[0], tf.random.uniform(shape=(), minval=0, maxval=1))
    return tf.logical_and(tf.math.greater(2.0, kd[0]), tf.math.greater(keep_prob[0], tf.random.uniform(shape=(), minval=0, maxval=1)))


def _filter_kds_deplete0(mir, image, kd, keep_prob, stype):
    outside1 = tf.math.greater(tf.abs(kd[0]), 1.0)
    coin_flip = tf.math.greater(1 / ((-3 * tf.math.abs(kd[0]) + 4) + 0.0001), tf.random.uniform(shape=(), minval=0, maxval=1))
    return tf.logical_or(outside1, coin_flip)


def _rev_comp(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq][::-1])


def _generate_random_seq(length):
    nts = ['A', 'T', 'C', 'G']
    seq = np.random.choice(nts, size=length, replace=True)
    return ''.join(seq)


def _get_target_no_match(mirna_sequence, length):
    """Given a miRNA sequence, return a random target sequence without 4 nt of contiguous pairing"""
    rc = _rev_comp(mirna_sequence[1:8]) + 'A'
    off_limits = [rc[ix:ix + 4] for ix in range(5)]
    while True:
        target = generate_random_seq(length)
        keep = True
        for subseq in off_limits:
            if subseq in target:
                keep = False
                break

        if keep:
            return target

