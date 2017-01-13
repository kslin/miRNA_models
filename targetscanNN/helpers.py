import numpy as np
import tensorflow as tf


def generate_random_sequence(length):
    """Generate a random RNA sequence of a given length"""
    
    nts = ['A','U','C','G']
    sequence = np.random.choice(nts, size=length, replace=True)

    return ''.join(sequence)


def get_complementary(seq):
    """Get the complementary sequence of a given RNA sequence"""
    
    intab = "AUCG"
    outtab = "UAGC"
    trantab = str.maketrans(intab, outtab)

    return seq.translate(trantab)


def generate_match_pair(length, random_seed=None):
    """Generate two sequences that are base-paired"""
    
    if random_seed is not None:
        np.random.seed(random_seed)

    seq1 = generate_random_sequence(length)
    seq2 = get_complementary(seq1)
    
    return seq1, seq2


def generate_seed_match_pair(length1, length2, random_seed=None):
    """Generate two sequences that are base-paired at positions 1-7"""
    
    if random_seed is not None:
        np.random.seed(random_seed)

    seq1 = generate_random_sequence(length1)
    up_fragment = generate_random_sequence(1)
    down_fragment = generate_random_sequence(length2-7)
    mid_fragment = get_complementary(seq1[1:7])
    
    seq2 = up_fragment + mid_fragment + down_fragment
    
    return seq1, seq2


def generate_random_pair(length1, length2, random_seed=None):
    """Generate two random sequences that are not perfectly complementary"""
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    seq1 = generate_random_sequence(length1)
    match_seq1 = get_complementary(seq1)

    while True:
        seq2 = generate_random_sequence(length2)

        if match_seq1 != seq2:
            return seq1, seq2


def one_hot_encode(seq, nt_order):
    """Convert RNA sequence to one-hot encoding"""
    
    one_hot = [list(np.array(nt_order == nt, dtype=int)) for nt in seq]
    one_hot = [item for sublist in one_hot for item in sublist]
    
    return np.array(one_hot)


def make_square(seq1, seq2):
    """Given two sequences, calculate outer product of one-hot encodings"""

    return np.outer(one_hot_encode(seq1, np.array(['A','U','C','G'])),
                    one_hot_encode(seq2, np.array(['U','A','G','C'])))

    
