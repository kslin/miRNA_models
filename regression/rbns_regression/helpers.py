import os
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
import regex
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
            heatmap = ax.pcolor(v, cmap=plt.cm.bwr, vmin=vmin, vmax=vmax, alpha=0.8)
            ax.set_frame_on(False)
            ax.set_xticks(np.arange(w) + 0.5, minor=False)
            ax.set_yticks(np.arange(h) + 0.5, minor=False)

            ax.invert_yaxis()
            ax.xaxis.tick_top()
            ax.set_xticklabels(xlabels, minor=False)
            ax.set_yticklabels(ylabels, minor=False)
            ax.grid(False)

            plot_num += 1

    plt.savefig(fname)
    plt.close()


## CLASSES ##

class Dataset(object):
    """create dataset object that holds features and labels and cycles through the data in batches"""

    def __init__(self, features, extra_features, labels):
        assert (len(features) == len(labels))
        self.features = np.array(features)
        self.extra_features = np.array(extra_features)
        self.labels = np.array(labels)
        self.index = 0
        self.size = len(labels)
    
    def next_batch(self, batch_size):
        old_index = self.index
        new_index = self.index + batch_size
        self.index = new_index % self.size
        if new_index <= self.size:
            return (self.features[old_index: new_index],
                    self.extra_features[old_index: new_index],
                    self.labels[old_index: new_index])
        else:
            subfeatures = np.concatenate([self.features[old_index:], self.features[:self.index]])
            subextra_features = np.concatenate([self.extra_features[old_index:], self.extra_features[:self.index]])
            sublabels = np.concatenate([self.labels[old_index:], self.labels[:self.index]])
            return (subfeatures, subextra_features, sublabels)
    
    def reset_index(self):
        self.index = 0


## Data Functions ##

def shuffle_file(infile, outfile):
    call(['sort','-R', '-o', outfile, infile])


def generate_random_seq(length):
    return ''.join(np.random.choice(['A','T','C','G'], size=length))

def complementaryT(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])

def get_color_old(sitem8, seq):
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


def get_color(sitem8, seq):
    if seq[2:-2] == (sitem8 + 'A'):
        return 'blue'
    elif seq[2:-3] == sitem8:
        return 'green'
    elif seq[3:-2] == (sitem8[1:] + 'A'):
        return 'orange'
    elif seq[3:-3] == (sitem8[1:]):
        return 'red'
    elif sitem8[1:] in seq:
        return 'offcenter'
    else:
        return 'grey'

def take_site(seq, site):
    if seq[7:10] != site[3:]:
        # print(seq, site)
        # sys.eixt()
        return False
    locs = regex.findall("({}){}".format(site, '{e<=2}'), seq)
    if len(locs) == 0:
        return False
    return True
    # for l in locs:
    #     if seq.index(l) == 3:
    #         return True
    # return False

def rev_comp(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])[::-1]

def complementary(seq):
    match_dict = {'A': 'U',
                  'U': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])

def complementaryT(seq):
    match_dict = {'A': 'T',
                  'T': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])

def one_hot_encode(c, classes):
    return list(np.array(classes == c, dtype=int))


def one_hot_encode_nt(seq, nt_order):
    """Convert RNA sequence to one-hot encoding"""
    
    one_hot = [list(np.array(nt_order == nt, dtype=int)) for nt in seq]
    one_hot = [item for sublist in one_hot for item in sublist]
    
    return np.array(one_hot)


def make_square(seq1, seq2):
    """Given two sequences, calculate outer product of one-hot encodings"""

    noise = np.random.normal(loc=0, scale=0.01, size=16*len(seq1)*len(seq2)).reshape((4*len(seq1), 4*len(seq2)))

    return np.outer(one_hot_encode_nt(seq1, np.array(['A','T','C','G'])),
                    one_hot_encode_nt(seq2, np.array(['T','A','G','C']))) + noise

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


def read_data(infile, len_data, dim1, dim2, extra=0, shuffle=True):
    """Reads in sequences and creates dataset objects"""

    # get length of data file
    len_features = dim1 * dim2
    

    # create arrays to hold data
    features = np.zeros((len_data, len_features))
    labels = np.zeros((len_data, 1))

    if extra > 0:
        extra_features = np.zeros((len_data, extra))

    else:
        extra_features = np.zeros((len_data, 0))
        extra_features.fill(None)

    # read in sequences and use 'make square' encoding
    with open(infile, 'r') as f:
        i = 0
        for line in f:
            line = line.split(',')
            features[i, :] = make_square(line[0],line[1]).flatten()
            labels[i,:] = [float(line[2])]
            if extra:
                extra_features[i, :] = [float(x) for x in line[3:3+extra]]
            i += 1

        assert (i == len_data)

    # shuffle data
    if shuffle:
        np.random.seed(0)
        ix = np.arange(len_data)
        np.random.shuffle(ix)

        features = features[ix, :]
        extra_features = extra_features[ix, :]
        labels = labels[ix, :]

    # create Dataset objects
    test_size = int(len(features)/10)
    train = Dataset(features[test_size:], extra_features[test_size:], labels[test_size:])
    test = Dataset(features[:test_size], extra_features[:test_size], labels[:test_size])

    return train, test


def read_flanking(infile, len_data, len_features, extra=0, shuffle=True):
    # create arrays to hold data
    features = np.zeros((len_data, len_features))
    labels = np.zeros((len_data, 1))

    if extra > 0:
        extra_features = np.zeros((len_data, extra))

    else:
        extra_features = np.zeros((len_data, 0))
        extra_features.fill(None)

    with open(infile, 'r') as f:
        i = 0
        for line in f:
            line = line.replace('\n','').split(',')
            labels[i, :] = [float(line[1])]
            feat = np.array([[int(x == nt) for nt in ['A','U','C','G']] for x in line[0]]).flatten()
            features[i, :] = feat
            if extra:
                extra_features[i, :] = [float(x) for x in line[2:]]
            i += 1

        assert (i == len_data)

    # shuffle data
    if shuffle:
        np.random.seed(0)
        ix = np.arange(len_data)
        np.random.shuffle(ix)

        features = features[ix, :]
        labels = labels[ix, :]

    # create Dataset objects
    test_size = int(len(features)/10)
    train = Dataset(features[test_size:], extra_features[test_size:], labels[test_size:])
    test = Dataset(features[:test_size], extra_features[:test_size], labels[:test_size])

    return train, test


def read_extras_only(infile, len_data, len_features, extra, shuffle=True):
    """Reads in extra features and creates dataset objects"""

    # create arrays to hold data
    features = np.zeros((len_data, len_features))
    extra_features = np.zeros((len_data, extra))
    labels = np.zeros((len_data, 1))

    # read in sequences and use 'make square' encoding
    with open(infile, 'r') as f:
        i = 0
        for line in f:
            line = line.split(',')
            labels[i,:] = [float(line[2])]
            extra_features[i, :] = [float(x) for x in line[3:]]
            i += 1

        assert (i == len_data)

    # shuffle data
    if shuffle:
        np.random.seed(0)
        ix = np.arange(len_data)
        np.random.shuffle(ix)

        features = features[ix, :]
        extra_features = extra_features[ix, :]
        labels = labels[ix, :]

    # create Dataset objects
    test_size = int(len(features)/10)
    train = Dataset(features[test_size:], extra_features[test_size:], labels[test_size:])
    test = Dataset(features[:test_size], extra_features[:test_size], labels[:test_size])

    return train, test


## NN Functions ##

def weight_variable(shape, n_in, name=None):
    # initial = tf.random_normal(shape, stddev=np.sqrt(2/n_in))
    initial = tf.truncated_normal(shape, stddev=0.1)
    # initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


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
    layer_name, padding='SAME', act=tf.nn.relu, init_weight=None, init_bias=None):
    """Create layer, given the input tensor, dimensions, and preactivation function
    """
    # add a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # create variables for weights and biases
        with tf.name_scope('weights'):
            if init_weight is None:
                weights = weight_variable([dim1, dim2, in_channels, out_channels], in_channels, name="{}_weight".format(layer_name))
            else:
                weights = tf.Variable(init_weight, name="{}_weight".format(layer_name))

            variable_summaries(weights)

            # add variable to collection of variables
            tf.add_to_collection('weight', weights)
        with tf.name_scope('biases'):
            if init_bias is None:
                biases = bias_variable([out_channels], name="{}_bias".format(layer_name))
            else:
                biases = tf.Variable(init_bias, name="{}_bias".format(layer_name))
            
            variable_summaries(biases)

            # add variable to collection of variables
            tf.add_to_collection('bias', biases)
            
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.nn.conv2d(input_tensor, weights, strides=[1, stride1, stride2, 1], padding=padding) + biases
            tf.summary.histogram('pre_activations', preactivate)
        
        out_layer = act(preactivate, name='activation')
        tf.summary.histogram('activations', out_layer)

        var_dict = {"{}_weight".format(layer_name): weights, "{}_bias".format(layer_name): biases}
        
        return out_layer, var_dict


def np_convert_diag(input_array):

    num, dim1, dim2, out_channels = input_array.shape
    mid = int(np.floor((dim1 + dim2 - 1)/2))

    output = np.random.randn(num, dim1 + dim2 - 1, dim2, out_channels) / 10.0

    for n in range(num):
        for i in range(dim1):
            for j in range(dim2):
                output[n, mid + (i-j), min(i,j), :] = input_array[n, i, j, :]


    return output

np_convert_diag_32 = lambda x: np_convert_diag(x).astype(np.float32)

def tf_convert_diag(x, name=None):
    with tf.name_scope(name, "convert_diag", [x]) as name:
        y = tf.py_func(np_convert_diag_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]

# x = np.array([[1,1,2,2,3,3,4,4],[5,5,6,6,7,7,8,8],[9,9,10,10,11,11,12,12]])

# with tf.Session() as sess:
#     blah1 = tf.placeholder(tf.float32, shape=[3, 8])
#     blah2 = tf.reshape(blah1, [3, 2, 2, 2])
#     blah3 = tf_convert_diag(blah2)

#     sess.run(tf.global_variables_initializer())


#     results = sess.run([blah2, blah3], feed_dict={blah1: x})

#     print(results)


def make_fullyconnected_layer(input_tensor, in_channels, out_channels, layer_name, act=tf.nn.relu):
    """Create layer, given the input tensor, dimensions, and preactivation function
    """
    # add a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # create variables for weights and biases
        with tf.name_scope('weights'):
            weights = weight_variable([in_channels, out_channels], in_channels)
            variable_summaries(weights)

            # add variable to collection of variables
            tf.add_to_collection('weight', weights)
        with tf.name_scope('biases'):
            biases = bias_variable([out_channels])
            variable_summaries(biases)

            # add variable to collection of variables
            tf.add_to_collection('bias', biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        
        out_layer = act(preactivate, name='activation')
        tf.summary.histogram('activations', out_layer)
        
        return out_layer 



def make_train_step_classification(tensor, y, starting_learning_rate):

    # global_step = tf.Variable(0, trainable=False) 
    # learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,
    #                                            decay_step, decay_rate, staircase=True)

    # tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tensor, y))
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(starting_learning_rate).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(tensor,1), tf.argmax(y,1))
    
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_step, accuracy, loss

def make_train_step_regression(loss_type, tensor, y, starting_learning_rate, data_weights=None):
    if loss_type == 'l2':
        with tf.name_scope('loss'):
            loss = tf.nn.l2_loss(tf.sub(tensor, y))
            # if data_weights is not None:
            #     loss = tf.reduce_sum(tf.multiply(tf.square(tf.sub(tensor, y)), data_weights))
            # else:
            #     loss = tf.nn.l2_loss(tf.sub(tensor, y))

    elif loss_type == 'poisson':
        with tf.name_scope('loss'):
            loss = tf.reduce_sum(tf.subtract(tf.exp(tensor), tf.multiply(tensor, y)))
    
    else:
        print('unknown loss_type, see function')

    SS_err = tf.reduce_sum(tf.square(tf.sub(tensor, y)))
    SS_tot = tf.reduce_sum(tf.square(tf.sub(y, tf.reduce_mean(y, 0))))

    with tf.name_scope('accuracy'):
        accuracy = tf.sub(tf.cast(1.0, tf.float32), tf.div(SS_err, SS_tot)) # R2

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(starting_learning_rate).minimize(loss)
        
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', loss)

    return train_step, accuracy, loss


# def fully_connected(tensor, weights, biases):
#     """Preactivation function for a fully connected layer"""

#     return tf.matmul(tensor, weights) + biases


# def convolution(tensor, weights, biases, stride1, stride2, padding='SAME'):
#     """Preactivation function for a fully connected layer"""

#     return tf.nn.conv2d(tensor, weights, strides=[1, stride1, stride2, 1], padding=padding) + biases
