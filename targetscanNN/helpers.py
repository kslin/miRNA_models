import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


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

def complementary(seq):
    match_dict = {'A': 'U',
                  'U': 'A',
                  'C': 'G',
                  'G': 'C'}

    return ''.join([match_dict[x] for x in seq])

def one_hot_encode(seq, nt_order):
    """Convert RNA sequence to one-hot encoding"""
    
    one_hot = [list(np.array(nt_order == nt, dtype=int)) for nt in seq]
    one_hot = [item for sublist in one_hot for item in sublist]
    
    return np.array(one_hot)


def make_square(seq1, seq2):
    """Given two sequences, calculate outer product of one-hot encodings"""

    return np.outer(one_hot_encode(seq1, np.array(['A','U','C','G'])),
                    one_hot_encode(seq2, np.array(['U','A','G','C'])))

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

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


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
    layer_name, padding='SAME', act=tf.nn.relu):
    """Create layer, given the input tensor, dimensions, and preactivation function
    """
    # add a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # create variables for weights and biases
        with tf.name_scope('weights'):
            weights = weight_variable([dim1, dim2, in_channels, out_channels])
            variable_summaries(weights)

            # add variable to collection of variables
            tf.add_to_collection('weight', weights)
        with tf.name_scope('biases'):
            biases = bias_variable([out_channels])
            variable_summaries(biases)

            # add variable to collection of variables
            tf.add_to_collection('bias', biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.nn.conv2d(input_tensor, weights, strides=[1, stride1, stride2, 1], padding=padding) + biases
            tf.summary.histogram('pre_activations', preactivate)
        
        out_layer = act(preactivate, name='activation')
        tf.summary.histogram('activations', out_layer)
        
        return out_layer


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
            weights = weight_variable([in_channels, out_channels])
            print(in_channels, out_channels)
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


def make_train_step(problem_type, tensor, y):
    if problem_type == 'classification':
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tensor, y))
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(tensor,1), tf.argmax(y,1))
        
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    elif problem_type == 'regression':
        SS_err = tf.reduce_sum(tf.square(tf.sub(tensor, y)))
        SS_tot = tf.reduce_sum(tf.square(tf.sub(y, tf.reduce_mean(y, 0))))
        R_2 = tf.sub(tf.cast(1.0, tf.float32), tf.div(SS_err, SS_tot))

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(SS_err)
        with tf.name_scope('accuracy'):
            accuracy = R_2

    elif problem_type == 'regression_l2':
        accuracy = tf.nn.l2_loss(tf.sub(tensor, y))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(accuracy)

    elif problem_type == 'regression_log_poisson':
        accuracy = tf.reduce_mean(tf.nn.log_poisson_loss(tensor, y, compute_full_loss=True))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(accuracy)

    
    else:
        print('problem_type must be \'classification\' or \'regression\'')
        
    tf.summary.scalar('accuracy', accuracy)

    return train_step, accuracy

# def fully_connected(tensor, weights, biases):
#     """Preactivation function for a fully connected layer"""

#     return tf.matmul(tensor, weights) + biases


# def convolution(tensor, weights, biases, stride1, stride2, padding='SAME'):
#     """Preactivation function for a fully connected layer"""

#     return tf.nn.conv2d(tensor, weights, strides=[1, stride1, stride2, 1], padding=padding) + biases
