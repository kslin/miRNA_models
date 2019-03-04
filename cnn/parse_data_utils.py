import numpy as np
import pandas as pd
import tensorflow as tf


def _parse_repression_function(serialized_example, parse_mirs, all_mirs, mirlen, seqlen, num_feats):
    """Parse the serialized example from tfrecords
    Inputs:
        serialized_example: input of reading tfrecords
        parse_mirs (list of strings): list of miRNAs to parse, need to explicity include * if parsing star strands
        parse_mirs (list of strings): list of all miRNAs used to generate tfrecords in the correct order
        mirlen (int): length of miRNA sequence encoded
        seqlen (int): length of site sequence encoded
        num_feats (int): number of additional features
    """

    # list of miRNA guide strands
    all_guides = [x for x in all_mirs if '*' not in x]

    # make masks for extracting only data from parse_mirs
    guide_mask = tf.constant(np.array([m in parse_mirs for m in all_guides]), dtype=tf.bool)
    mir_mask = tf.constant(np.array([m in parse_mirs for m in all_mirs]), dtype=tf.bool)

    # construct feature descriptions
    feature_description = {
        'transcript': tf.FixedLenFeature([], tf.string, default_value=''),
        'tpms': tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'nsites': tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
    }
    for mir in parse_mirs:
        feature_description['{}_mir_1hot'.format(mir)] = tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)
        feature_description['{}_seqs_1hot'.format(mir)] = tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)
        feature_description['{}_ts7_features'.format(mir)] = tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)

    parsed = tf.parse_single_example(serialized_example, feature_description)
    tpms = tf.expand_dims(tf.boolean_mask(parsed['tpms'], guide_mask), axis=0)
    nsites = tf.boolean_mask(parsed['nsites'], mir_mask)
    images = []
    ts7_features = []
    for ix, mir in enumerate(parse_mirs):

        # perform outer product between miRNA 1hot and site 1hot, reshape to [num_sites, 4 * mirlen, 4 * seqlen]
        image = tf.tensordot(parsed['{}_mir_1hot'.format(mir)], parsed['{}_seqs_1hot'.format(mir)], axes=0)
        image = tf.transpose(tf.reshape(image, [4 * mirlen, -1, 4 * seqlen]), perm=[1, 0, 2])
        images.append(image)

        # reshape features to [num_sites, num_ts7_features]
        ts7_features.append(tf.reshape(parsed['{}_ts7_features'.format(mir)], [-1, num_feats]))

    images = tf.concat(images, axis=0)
    ts7_features = tf.concat(ts7_features, axis=0)

    # return parsed
    return images, ts7_features, tpms, tf.cast(nsites, tf.int32), parsed['transcript']


def _build_tpm_batch(iterator, batch_size):
    images, features, labels, nsites, transcripts = [], [], [], [], []
    for _ in range(batch_size):
        results = iterator.get_next()
        images.append(results[0])
        features.append(results[1])
        labels.append(results[2])
        nsites.append(results[3])
        transcripts.append(results[4])

    results = {
        'images': tf.concat(images, axis=0),
        'features': tf.concat(features, axis=0),
        'labels': tf.concat(labels, axis=0),
        'nsites': tf.concat(nsites, axis=0),
        'transcripts': tf.stack(transcripts)
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
        'log_kd': tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)
    }

    parsed = tf.parse_single_example(serialized_example, feature_description)
    image = tf.tensordot(parsed['mir_1hot'], parsed['seq_1hot'], axes=0)

    # return parsed data
    return parsed['mir'], image, parsed['log_kd']


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

