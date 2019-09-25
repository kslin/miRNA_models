import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


def plot_weights(cnn_weights, logdir, mirlen, seqlen):
    # plot weights
    conv_weights = cnn_weights['w1']
    xlabels = ['A', 'U', 'C', 'G']
    ylabels = ['A', 'U', 'C', 'G']
    graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(logdir, 'convolution1.pdf'))

    # plot importance matrix
    conv_weights = np.abs(cnn_weights['w3'])
    conv_weights = np.sum(conv_weights, axis=(2, 3))
    vmin, vmax = np.min(conv_weights), np.max(conv_weights)
    xlabels = ['s{}'.format(i + 1) for i in range(seqlen)]
    ylabels = ['m{}'.format(i + 1) for i in list(range(mirlen))[::-1]]
    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(conv_weights, xticklabels=xlabels, yticklabels=ylabels,
                cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
    plt.savefig(os.path.join(logdir, 'convolution3.pdf'))
    plt.close()


def plot_scalars(scalar_dict, logdir):
    for key, val in scalar_dict.items():
        if len(val) > 0:
            fig = plt.figure(figsize=(7, 5))
            plt.plot(val)
            plt.savefig(os.path.join(logdir, '{}.png'.format(key)))
            plt.close()
