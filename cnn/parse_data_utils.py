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


def parse_sites_function(serialized_example, num_mirs, mirlen, seqlen):
    # construct feature descriptions
    transcript_features = {
        'transcript': tf.FixedLenFeature([], tf.string, default_value=''),
        'batch': tf.FixedLenFeature([], tf.int64, default_value=0),
    }

    site_features = {
        'mir': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'tpm': tf.FixedLenSequenceFeature([], dtype=tf.float32),
        # 'mirseq_1hot': tf.FixedLenSequenceFeature([mirlen], dtype=tf.int64),
        'orf_guide_1hot': tf.VarLenFeature(dtype=tf.int64),
        'utr3_guide_1hot': tf.VarLenFeature(dtype=tf.int64),
        # 'mirseq*_1hot': tf.FixedLenSequenceFeature([mirlen], dtype=tf.int64),
        'orf_pass_1hot': tf.VarLenFeature(dtype=tf.int64),
        'utr3_pass_1hot': tf.VarLenFeature(dtype=tf.int64),
    }

    # Extract features from serialized data
    transcript_data, site_data = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=transcript_features,
        sequence_features=site_features)

    # elems = (site_data['mirseq_1hot'], site_data['orf_siteseq_1hot'])
    guide_orf_image = site_data['orf_guide_1hot']

    return transcript_data['batch'], guide_orf_image


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

    # # list of miRNA guide strands
    # all_guides = [x for x in all_mirs if '*' not in x]

    # # make masks for extracting only data from parse_mirs
    # guide_mask = tf.constant(np.array([m in parse_mirs for m in all_guides]), dtype=tf.bool)
    # mir_mask = tf.constant(np.array([m in parse_mirs for m in all_mirs]), dtype=tf.bool)
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
        # 'tpms': tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        # 'guides': tf.FixedLenSequenceFeature([], tf.string, default_value='', allow_missing=True),
        # 'mirs': tf.FixedLenSequenceFeature([], tf.string, default_value='', allow_missing=True),
        # 'nsites': tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
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
    # guide_mask = tf.where(parsed['guides'] not in parse_mirs)
    # mir_mask = tf.where(parsed['mirs'] not in parse_mirs)
    # tpms = tf.expand_dims(tf.boolean_mask(parsed['tpms'], guide_mask), axis=0)
    # nsites = tf.boolean_mask(parsed['nsites'], mir_mask)
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
    return images, ts7_features, tpms, tf.cast(nsites, tf.int32), parsed['transcript'], parsed['batch']


def _build_tpm_batch(iterator, batch_size):
    images, features, labels, nsites, transcripts, batches = [], [], [], [], [], []
    for ix in range(batch_size):
        results = iterator.get_next()

        images.append(results[0])
        features.append(results[1])
        labels.append(results[2])
        nsites.append(results[3])
        transcripts.append(results[4])
        batches.append(results[5])

    results = {
        'images': tf.concat(images, axis=0),
        'features': tf.concat(features, axis=0),
        'labels': tf.stack(labels, axis=0),
        'nsites': tf.concat(nsites, axis=0),
        'transcripts': tf.stack(transcripts),
        'batches': tf.stack(batches),
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


def _filter_kds_train(mir, image, kd, keep_prob, stype):
    better_than_nosite = tf.math.less(kd[0], 0.0)
    # return better_than_nosite
    return tf.logical_and(better_than_nosite, tf.math.greater(keep_prob[0], tf.random.uniform(shape=(), minval=0, maxval=1)))
    # return tf.math.greater(keep_prob[0], tf.random.uniform(shape=(), minval=0, maxval=1))
    
    
    # keep_withsite = tf.math.greater(keep_prob[0], tf.random.uniform(shape=(), minval=0, maxval=1))
    # keep_withsite = tf.logical_and(better_than_nosite, keep_withsite)
    # keep_nosite = tf.math.greater(keep_prob[0], tf.random.uniform(shape=(), minval=0, maxval=1))
    # keep_nosite = tf.logical_and(tf.logical_not(better_than_nosite), keep_nosite)
    # # return keep
    # return tf.logical_or(keep_withsite, keep_nosite)
    # return tf.math.greater(tf.sigmoid((-1.0 * kd) - 3.0), tf.random.uniform(shape=(), minval=0, maxval=1))[0]
    # return tf.math.greater(tf.sigmoid((-1.0 * kd) - 2.0), tf.constant([0.5]))[0]



def _filter_kds_val(mir, image, kd, keep_prob, stype):
    return tf.math.greater(keep_prob[0], tf.random.uniform(shape=(), minval=0, maxval=1))
#     better_than_nosite = tf.math.less(kd[0], -2.0)
    
#     keep_withsite = tf.math.greater(keep_prob[0], tf.random.uniform(shape=(), minval=0, maxval=1))
#     keep_withsite = tf.logical_and(better_than_nosite, keep_withsite)
#     return keep_withsite


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

