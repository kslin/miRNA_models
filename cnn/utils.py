import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def rev_comp(seq):
    """ Get reverse complement """

    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq][::-1])


def one_hot_encode(seq):
    """ 1-hot encode ATCG sequence """
    nt_dict = {
        'A': 0,
        'T': 1,
        'C': 2,
        'G': 3,
        'X': 4
    }
    targets = np.zeros([5, 4])
    targets[:4, :] = np.eye(4)

    seq = [nt_dict[nt] for nt in seq]
    return targets[seq].flatten().astype(int)


def generate_random_seq(length):
    """Generate random sequence"""
    nts = ['A', 'T', 'C', 'G']
    seq = np.random.choice(nts, size=length, replace=True)
    return ''.join(seq)


def get_target_no_match(mirna_sequence, length):
    """Given a miRNA sequence, return a random target sequence without 4 nt of contiguous pairing"""
    rc = rev_comp(mirna_sequence[1:8]) + 'A'
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

def get_mir_no_match(site_sequence, length):
    """Given a target sequence, return a random miRNA sequence without 4 nt of contiguous pairing"""
    rc = rev_comp(site_sequence[2:-3])
    off_limits = [rc[ix:ix + 4] for ix in range(4)]
    while True:
        target = generate_random_seq(length)
        keep = True
        for subseq in off_limits:
            if subseq in target:
                keep = False
                break

        if keep:
            return target

### GRAPHING FUNCTIONS ###

def graph_convolutions(conv_weights, xlabels, ylabels, fname):
    vmin, vmax = np.min(conv_weights), np.max(conv_weights)
    dim = conv_weights.shape
    nrows = dim[2]
    ncols = dim[3]
    h, w = dim[0], dim[1]

    if xlabels is None:
        xlabels = [str(x) for x in (np.arange(w) + 1)[::-1]]

    if ylabels is None:
        ylabels = [str(y) for y in (np.arange(h) + 1)[::-1]]

    plot_num = 1
    fig = plt.figure(figsize=(w * ncols, h * nrows))
    for i in range(nrows):
        for j in range(ncols):
            v = conv_weights[:, :, i, j].reshape(h, w)
            ax = plt.subplot(nrows, ncols, plot_num)
            sns.heatmap(v, xticklabels=xlabels, yticklabels=ylabels,
                        cmap=plt.cm.bwr, vmin=vmin, vmax=vmax, ax=ax)
            plot_num += 1

    # plt.colorbar()
    plt.tight_layout()
    fig.savefig(fname)
    plt.close()
