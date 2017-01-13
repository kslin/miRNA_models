import sys

import numpy as np
import tensorflow as tf

import helpers
import classes

if __name__ == '__main__':

    INFILE, LEN_DATA, LEN_FEATURES, LOGDIR = sys.argv[1:]
    LEN_DATA, LEN_FEATURES = int(LEN_DATA), int(LEN_FEATURES)

    print('Reading in data')

    features = np.zeros((LEN_DATA, LEN_FEATURES))
    labels = np.zeros((LEN_DATA, 1))

    with open(INFILE, 'r') as f:
        for i,line in enumerate(f):
            line = line.split(',')
            features[i, :] = helpers.make_square(line[0],line[1]).flatten()
            labels[i,:] = [float(line[2])]

    print('Shuffling data')

    ix = np.arange(LEN_DATA)
    np.random.shuffle(ix)

    features = features[ix, :]
    labels = labels[ix, :]

    test_size = int(len(features)/10)
    constructed_logfc_train = classes.Dataset(features[test_size:], labels[test_size:])
    constructed_logfc_test = classes.Dataset(features[:test_size], labels[:test_size])

    print('Training neural network')

    with tf.Session() as sess:

        dim1, dim2 = 80, 96
        num_output1 = 16
        num_output2 = 16
        fully_connected_nodes = 32
        out_nodes = 1

        NN = classes.NeuralNet2D(sess, dim1, dim2, out_nodes)
        NN.add_convolution('convolution1', 4, 4, 4, 4, num_output1)
        NN.add_convolution('convolution2', 2, 2, 1, 1, num_output2)
        NN.add_fully_connected('fullyconnected', fully_connected_nodes)
        NN.add_dropout('dropout', out_nodes)
        NN.make_train_step('regression', LOGDIR)
        NN.train_model(constructed_logfc_train, constructed_logfc_test, num_epoch=10000,
                       batch_size=100, report_int=1000, keep_prob_train=0.5)

        # dim1, dim2 = 16, 16
        # num_output1 = 4
        # fully_connected_nodes = 1024
        # out_nodes = 2

        # NN = classes.NeuralNet2D(sess, dim1, dim2, out_nodes)
        # NN.add_convolution('convolution1', 4, 4, 4, 4, num_output1)
        # NN.add_fully_connected('fullyconnected1', fully_connected_nodes)
        # NN.add_dropout('dropout', out_nodes)
        # NN.make_train_step('classification', '../../logdirs/match')
        # NN.train_model(match_train, match_test, num_epoch=1000,
        #                batch_size=50, report_int=10, keep_prob_train=0.5)
