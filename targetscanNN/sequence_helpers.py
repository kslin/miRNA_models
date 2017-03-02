import sys

import numpy as np

import helpers

def get_random_sequence(length):
    nts = ['A','U','C','G']
    return ''.join(np.random.choice(nts, size=length, replace=True))


def get_match(sequence):
    match_dict = {'A':'U', 'U':'A',
                  'C':'G', 'G':'C'}

    return ''.join([match_dict[char] for char in sequence])


def generate_sample_data1(num_mir, size, mir_length, seq_length):

    # generate random mirna sequences
    mirs = [get_random_sequence(mir_length) for i in range(num_mir)]

    features = np.zeros((size, 16*mir_length*seq_length))
    extra_features = np.zeros((size, 0))
    labels = np.zeros((size, 2))
    i = 0
    for i in range(size):
        mir = np.random.choice(mirs)
        if np.random.random() < 0.5:
            seq = get_random_sequence(seq_length)
            labels[i,:] = [0,1]
        else:
            seq = get_random_sequence(1) + get_match(mir[1:7]) + get_random_sequence(seq_length-7)
            labels[i,:] = [1,0]

        features[i,:] = helpers.make_square(mir, seq).flatten()

    print(np.sum(labels,0))


    # create Dataset objects
    test_size = int(len(features)/10)
    train = helpers.Dataset(features[test_size:], extra_features[test_size:], labels[test_size:])
    test = helpers.Dataset(features[:test_size], extra_features[:test_size], labels[:test_size])

    return train, test


def generate_sample_data(num_mir, size, mir_length, seq_length):

    # generate random mirna sequences
    mirs = [get_random_sequence(mir_length) for i in range(num_mir)]

    features = np.zeros((size, 16*mir_length*seq_length))
    extra_features = np.zeros((size, 0))
    labels = np.zeros((size, 2))
    i = 0
    while i < size:
        mir = np.random.choice(mirs)
        if np.random.random() < 0.5:
            before = np.random.randint(seq_length - 6)
            after = seq_length - 6 - before
            seq = get_random_sequence(before) + get_match(mir[1:7]) + get_random_sequence(after)
        else:
            seq = get_random_sequence(seq_length)
            if (get_match(mir[1:7]) in seq):
                continue
        #     if np.random.random() < 0.5:
        #         continue
        features[i,:] = helpers.make_square(mir, seq).flatten()
        if (get_match(mir[1:7]) in seq):
            labels[i,:] = [0,1]
        else:
            labels[i,:] = [1,0]

        i += 1


    print(np.sum(labels,0))


    # create Dataset objects
    test_size = int(len(features)/10)
    train = helpers.Dataset(features[test_size:], extra_features[test_size:], labels[test_size:])
    test = helpers.Dataset(features[:test_size], extra_features[:test_size], labels[:test_size])

    return train, test

# generate_sample_data(1, 10, 23, 40)
