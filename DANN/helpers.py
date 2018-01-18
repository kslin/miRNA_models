import os
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
import regex
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework import ops


## GRAPHING ##
def graph_predicted_v_actual(ncols, out_nodes, predicted, actual, colors, fname, log_scale=False):
    fig = plt.figure(figsize=(5*ncols,10))
    for i in range(out_nodes):
        ax = plt.subplot(2,ncols,i+1)
        if log_scale:
            ax.set_yscale("log")
            predicted += 1
            actual += 1
        ax.scatter(predicted[:, i], actual[:, i], c=colors, alpha=0.5, s=20)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


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
    fig = plt.figure(figsize=(w*ncols, h*nrows))
    for i in range(nrows):
        for j in range(ncols):
            v = conv_weights[:,:,i,j].reshape(h,w)
            ax = plt.subplot(nrows, ncols, plot_num)
            sns.heatmap(v, xticklabels=xlabels, yticklabels=ylabels,
                        cmap=plt.cm.bwr, vmin=vmin, vmax=vmax)
            # heatmap = ax.pcolor(v, cmap=plt.cm.bwr, , alpha=0.8)
            # ax.set_frame_on(False)
            # ax.set_xticks(np.arange(w) + 0.5, minor=False)
            # ax.set_yticks(np.arange(h) + 0.5, minor=False)

            # ax.invert_yaxis()
            # ax.xaxis.tick_top()
            # ax.set_xticklabels(xlabels, minor=False)
            # ax.set_yticklabels(ylabels, minor=False)
            # ax.grid(False)

            plot_num += 1

    # plt.colorbar()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


## Data Functions ##

def generate_random_seq(length):
    return ''.join(np.random.choice(['A','T','C','G'], size=length, replace=True))

def get_color(sitem8, seq):
    if (sitem8 + 'A') in seq:
        return 'blue'
    elif sitem8 in seq:
        return 'green'
    elif (sitem8[1:] + 'A') in seq:
        return 'orange'
    elif (sitem8[1:]) in seq:
        return 'red'
    else:
        return 'grey'

def rev_comp(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])[::-1]


def complementary(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])


def one_hot_encode_nt(seq, nt_order):
    """Convert RNA sequence to one-hot encoding"""
    
    one_hot = [list(np.array(nt_order == nt, dtype=int)) for nt in seq]
    one_hot = [item for sublist in one_hot for item in sublist]
    
    return np.array(one_hot)


def make_square(seq1, seq2):
    """Given two sequences, calculate outer product of one-hot encodings"""

    # noise = np.random.normal(loc=0, scale=0.01, size=16*len(seq1)*len(seq2)).reshape((4*len(seq1), 4*len(seq2)))

    square = np.outer(one_hot_encode_nt(seq1, np.array(['A','T','C','G'])),
                    one_hot_encode_nt(seq2, np.array(['T','A','G','C'])))

    square = (square*4) - 0.25

    return square# + noise

def make_cube(seq1, seq2):
    nt_order1 = np.array(['A','T','C','G'])
    nt_order2 = np.array(['T','A','G','C'])
    one_hot1 = np.array([[list(np.array(nt_order1 == nt, dtype=int)) for nt in seq1]]).reshape(len(seq1),1,4,1)
    one_hot2 = np.array([[list(np.array(nt_order2 == nt, dtype=int)) for nt in seq2]]).reshape(1,len(seq2),1,4)
    
    return np.matmul(one_hot1, one_hot2)

def get_file_length(filename):
    i = 1
    first_line = ""
    with open(filename, 'r') as f:
        first_line = f.readline()
        for line in f:
            i += 1

    first_line = first_line.split(',')
    dim1, dim2 = len(first_line[0]) * 4, len(first_line[1]) * 4

    return i, dim1, dim2


## NN Functions ##

def weight_variable(shape, init_std=0.1, name=None):
    var = tf.get_variable(
                            name=name,
                            initializer=tf.random_normal_initializer(stddev=init_std),
                            shape=shape
    )
    return var


def bias_variable(shape, init=0.01, name=None):
    var = tf.get_variable(
                            name=name,
                            initializer=tf.constant_initializer(init),
                            shape=shape
    )
    return var


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def make_convolution_layer(input_tensor, dim1, dim2, in_channels, out_channels, stride1, stride2,
    layer_name, phase_train, padding='SAME', act=tf.nn.relu):
    """Create layer, given the input tensor, dimensions, and preactivation function
    """
    # add a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # create variables for weights and biases
        with tf.name_scope('weights'):
            weights = weight_variable([dim1, dim2, in_channels, out_channels], name="{}_weight".format(layer_name))
            variable_summaries(weights)

            # add variable to collection of variables
            tf.add_to_collection('weight', weights)
        with tf.name_scope('biases'):
            biases = bias_variable([out_channels], name="{}_bias".format(layer_name))
            variable_summaries(biases)

            # add variable to collection of variables
            tf.add_to_collection('bias', biases)
            
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.nn.conv2d(input_tensor, weights, strides=[1, stride1, stride2, 1], padding=padding) + biases
            tf.summary.histogram('pre_activations', preactivate)

        bn_layer = tf.layers.batch_normalization(preactivate, axis=1, training=phase_train, trainable=True, name=layer_name)
        
        out_layer = act(bn_layer, name='activation')
        tf.summary.histogram('activations', out_layer)

        var_dict = {"{}_weight".format(layer_name): weights,
                    "{}_bias".format(layer_name): biases}
        
        return out_layer, var_dict


def make_fullyconnected_layer(input_tensor, in_channels, out_channels, layer_name, act=tf.nn.relu):
    """Create layer, given the input tensor, dimensions, and preactivation function
    """
    # add a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # create variables for weights and biases
        with tf.name_scope('weights'):
            weights = weight_variable([in_channels, out_channels], name="{}_weight".format(layer_name))
            variable_summaries(weights)

            # add variable to collection of variables
            tf.add_to_collection('weight', weights)
        with tf.name_scope('biases'):
            biases = bias_variable([out_channels], name="{}_bias".format(layer_name))
            variable_summaries(biases)

            # add variable to collection of variables
            tf.add_to_collection('bias', biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        
        out_layer = act(preactivate, name='activation')
        tf.summary.histogram('activations', out_layer)

        var_dict = {"{}_weight".format(layer_name): weights, "{}_bias".format(layer_name): biases}
        
        return out_layer, var_dict



# def make_train_step_classification(tensor, y, starting_learning_rate):

#     # global_step = tf.Variable(0, trainable=False) 
#     # learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,
#     #                                            decay_step, decay_rate, staircase=True)

#     # tf.summary.scalar('learning_rate', learning_rate)

#     with tf.name_scope('loss'):
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tensor, y))
#     with tf.name_scope('train'):
#         train_step = tf.train.AdamOptimizer(starting_learning_rate).minimize(loss)
    
#     correct_prediction = tf.equal(tf.argmax(tensor,1), tf.argmax(y,1))
    
#     with tf.name_scope('accuracy'):
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#     return train_step, accuracy, loss

# def make_train_step_regression(tensor, y, starting_learning_rate, weight_regularize, data_weights=None):
#     extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(extra_update_ops):
#         with tf.name_scope('loss'):
#             loss = tf.nn.l2_loss(tf.subtract(tensor, y)) + weight_regularize

#         SS_err = tf.reduce_sum(tf.square(tf.subtract(tensor, y)))
#         SS_tot = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y, 0))))

#         with tf.name_scope('accuracy'):
#             accuracy = tf.subtract(tf.cast(1.0, tf.float32), tf.divide(SS_err, SS_tot)) # R2

#         with tf.name_scope('train'):
#             train_step = tf.train.AdamOptimizer(starting_learning_rate).minimize(loss)
            
#         tf.summary.scalar('accuracy', accuracy)
#         tf.summary.scalar('loss', loss)

#     return train_step, accuracy, loss