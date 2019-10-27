import itertools as it
import numpy as np


def generate_12mers(site8, only_canon):
    mers = []
    if only_canon:
        all_6mers = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"], repeat=6))]
        for i in range(3):
            subseq = site8[i:i+6]
            mers += [x[:i+2] + subseq + x[i+2:] for x in all_6mers]

    else:
        all_8mers = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"], repeat=8))]
        for i in range(5):
            subseq = site8[i:i+4]
            mers += [x[:i+2] + subseq + x[i+2:] for x in all_8mers]
    
    mers = list(set(mers))
    return sorted(mers)


def rev_comp(seq):
    """
    Parameters:
    ==========
    seq: string, sequence to get reverse complement of

    Returns:
    =======
    float: reverse complement of seq in all caps
    """
    seq = seq.upper()
    intab = b'ATCG'
    outtab = b'TAGC'
    trantab = bytes.maketrans(intab, outtab)
    seq = seq[::-1]
    seq = seq.translate(trantab)
    return seq


def get_centered_stype(site8, seq):
    if site8 == seq[2:-2]:
        return '8mer'
    elif site8[:-1] == seq[2:-3]:
        return '7mer-m8'
    elif site8[1:] == seq[3:-2]:
        return '7mer-a1'
    elif site8[1:-1] == seq[3:-3]:
        return '6mer'
    elif site8[:-2] == seq[2:-4]:
        return '6mer-m8'
    elif site8[2:] == seq[4:-2]:
        return '6mer-a1'
    else:
        return 'no site'


def generate_random_seq(length):
    """Generate random sequence"""
    nts = ['A', 'T', 'C', 'G']
    seq = np.random.choice(nts, size=length, replace=True)
    return ''.join(seq)
