from optparse import OptionParser
import sys

import numpy as np
import pandas as pd
import regex
from sklearn.linear_model import LinearRegression
import tensorflow as tf

import helpers
import sequence_helpers


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-i", "--train_file", dest="TRAIN_FILE", help="training data")
    parser.add_option("-t", "--test_file", dest="TEST_FILE", help="testing data")
    parser.add_option("-m", "--mirlen", dest="MIRLEN", type="int", help="length of miRNA")
    parser.add_option("-s","--seqlen", dest="SEQLEN", type="int", help="length of sequence")
    parser.add_option("-d","--datalen", dest="DATALEN", type="int", help="length of data")
    parser.add_option("-e","--extra", default=0, type="int", dest="NUM_EXTRA", help="number of extra features to use")
    parser.add_option("-l","--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("-y", "--summary",
                      action="store_true", dest="SUMMARY", default=False,
                      help="write summaries for tensorboard")

    (options, args) = parser.parse_args()

    print('Building neural network')

    with tf.Session() as sess:

        out_nodes = 7

        # create placeholders for data
        x = tf.placeholder(tf.float32, shape=[None, 16 * options.MIRLEN * options.SEQLEN])
        y = tf.placeholder(tf.float32, shape=[None, out_nodes])
        x_image = tf.reshape(x, [-1, 4*options.MIRLEN, 4*options.SEQLEN, 1])
        
        # create summaries of features for visualizing with tensorboard
        x_summary = tf.summary.image('images', x_image)

        ## LAYER 1 ##

        # convolution layer for each 4x4 box representing two paired nucleotides
        in_channels = 1
        out_channels = 4
        layer1 = helpers.make_convolution_layer(x_image, 4, 4, in_channels, out_channels,
                                                4, 4, 'conv4x4', padding='SAME', act=tf.nn.relu)

        # convert to diagonal shape
        # layer1_diag = helpers.tf_convert_diag(layer1)

        # print(layer1_diag.get_shape().as_list())

        ## LAYER 2 ##

        # convolution layer for each 4x4 box
        in_channels = out_channels
        out_channels = 8
        layer2_pre = helpers.make_convolution_layer(layer1, 4, 4, in_channels, out_channels,
                                                  1, 1, 'convlayer2', padding='SAME', act=tf.nn.relu)

        # in_channels = out_channels
        # out_channels = 8
        # layer2_1 = helpers.make_convolution_layer(layer2, 4, 4, in_channels, out_channels,
        #                                           1, 1, 'convlayer3', padding='VALID', act=tf.nn.relu)

        # print(layer2.get_shape().as_list())
        layer2 = tf.nn.max_pool(layer2_pre, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        ## LAYER 2.1 ##
        # in_channels = out_channels
        # out_channels = 8
        # layer2_1_pre = helpers.make_convolution_layer(layer2, 5, 5, in_channels, out_channels,
        #                                           1, 1, 'conv4x4', padding='SAME', act=tf.nn.relu)
        # layer2_1 = tf.nn.max_pool(layer2_1_pre, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # reshape to 1D tensor
        layer_flat_dim = layer2.get_shape().as_list()
        print(layer_flat_dim)
        layer_flat_dim = layer_flat_dim[1] * layer_flat_dim[2] * layer_flat_dim[3]
        print(layer_flat_dim)
        layer_flat = tf.reshape(layer2, [-1, layer_flat_dim])

        ## LAYER 3 ##

        # add fully connected layer
        in_channels = layer_flat_dim
        out_channels = 64
        layer3 = helpers.make_fullyconnected_layer(layer_flat, in_channels, out_channels, 'fullyconnected1', act=tf.nn.relu)

        # add dropout
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropout = tf.nn.dropout(layer3, keep_prob)

        # # add in extra features
        # dropout_plus_extra = tf.concat(1, [dropout, extra])

        # ## LAYER 4 ##

        # add last layer with 1 output channel
        # in_channels = out_channels + options.NUM_EXTRA
        in_channels = out_channels
        # in_channels = layer_flat_dim
        out_channels = out_nodes
        layer4 = helpers.make_fullyconnected_layer(dropout, in_channels, out_channels, 'fullyconnected2', act=tf.identity)

        # add extra features
        # layer4 = tf.concat(1, [layer4, extra])



        ## LAYER 5 ##

        # add last layer with 1 output channel
        # in_channels = options.NUM_EXTRA
        # out_channels = out_nodes
        # layer5 = helpers.make_fullyconnected_layer(extra, in_channels, out_channels, 'fullyconnected3', act=tf.identity)


        # prepare training steps and log writers
        train_step, accuracy = helpers.make_train_step('regression', layer4, y)

        merged = tf.summary.merge_all()

        if options.SUMMARY:
            train_writer = tf.summary.FileWriter(options.LOGDIR + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(options.LOGDIR + '/test')


        mir = 'UGAGGUAGUAGGUUGUAUGGUU'
        mir = mir[:options.MIRLEN][::-1].replace('U','T')
        seed_site = 'TACCTC'

        print('Reading test data')
        data = pd.read_csv(options.TEST_FILE, header=None)
        # data[0] = [x.replace('T','U') for x in data[0]]
        data['use'] = [int(len(regex.findall("({}){}".format(seed_site, '{e<=1}'), seq)) > 0) for seq in data[0]]
        print(len(data))
        data = data[data['use'] == 1]
        print(len(data))
        # test_labels = data[list(range(1,8))].values.astype(float) + 1
        test_labels = np.log(data[list(range(1,8))].values.astype(float) + 1)

        test_features = np.zeros((len(data), 16 * options.MIRLEN * options.SEQLEN))
        for i, seq in enumerate(list(data[0])):
            test_features[i,:] = helpers.make_square(mir, seq).flatten()


        print('Training model')

        # TRAIN MODEL #
        num_epoch = 150000
        batch_size = 100
        report_int = 1000
        keep_prob_train = 0.5

        # create a saver for saving the model after training
        saver = tf.train.Saver()

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # train epochs and record performance
        sample_counter = 0
        batch_counter = 0
        line_counter = 0
        features = np.zeros((batch_size, 16 * options.MIRLEN * options.SEQLEN))
        labels = np.zeros((batch_size, out_nodes))
        with open(options.TRAIN_FILE, 'r') as infile:
            for line in infile:
                if line_counter >= (options.DATALEN - 1):
                    line_counter = 0
                    infile.seek(0)
                if sample_counter == batch_size:
                    
                    if batch_counter%report_int == 0:
                
                        acc, summary = sess.run([accuracy, merged], feed_dict={x: test_features,
                                                                               keep_prob: 1.0,
                                                                               y: test_labels})

                        if options.SUMMARY:
                            test_writer.add_summary(summary, batch_counter)

                        print('Accuracy at step %s: %s' % (batch_counter, acc))

                        # save model
                        saver.save(sess, options.LOGDIR + '/my-model', global_step=batch_counter)
            
                    else:
                        _, summary = sess.run([train_step, merged], feed_dict={x: features,
                                                                               keep_prob: keep_prob_train,
                                                                               y: labels})

                        if options.SUMMARY:
                            train_writer.add_summary(summary, batch_counter)


                    batch_counter += 1
                    sample_counter = 0
                    features = np.zeros((batch_size, 16 * options.MIRLEN * options.SEQLEN))
                    labels = np.zeros((batch_size, out_nodes))
                    if batch_counter == num_epoch:
                        break

                vals = line.replace('\n','').split(',')
                seq = vals[0]
                # seq = vals[0].replace('T','U')
                # if ('UACCUC' in seq):
                if len(regex.findall("({}){}".format(seed_site, '{e<=1}'), seq)) > 0:
                    features[sample_counter, :] = helpers.make_square(mir, seq).flatten()
                    # labels[sample_counter, :] = [float(x) + 1 for x in vals[1:]]
                    labels[sample_counter, :] = [np.log(float(x) + 1) for x in vals[1:]]
                    sample_counter += 1
                line_counter += 1


        print(batch_counter, num_epoch, line_counter)

