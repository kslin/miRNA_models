from optparse import OptionParser
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf

import helpers


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="INFILE", help="FILE with features")
    parser.add_option("-l","--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("-a","--accuracyfile", dest="ACCFILE", help="file for writing accuracies")
    parser.add_option("-e","--extra", default=0, type="int", dest="NUM_EXTRA", help="number of extra features to use")
    parser.add_option("-s", "--summary",
                      action="store_true", dest="SUMMARY", default=False,
                      help="write summaries for tensorboard")

    (options, args) = parser.parse_args()

    LEN_DATA, DIM1, DIM2 = helpers.get_file_length(options.INFILE)

    print('Building neural network')

    with tf.Session() as sess:

        out_nodes = 1

        # create placeholders for data
        x = tf.placeholder(tf.float32, shape=[None, DIM1*DIM2])
        extra = tf.placeholder(tf.float32, shape=[None, options.NUM_EXTRA])
        y = tf.placeholder(tf.float32, shape=[None, out_nodes])
        x_image = tf.reshape(x, [-1, DIM1, DIM2, 1])
        
        # create summaries of features for visualizing with tensorboard
        x_summary = tf.summary.image('images', x_image)

        ## LAYER 1 ##

        # convolution layer for each 4x4 box representing two paired nucleotides
        in_channels = 1
        out_channels = 4
        layer1 = helpers.make_convolution_layer(x_image, 4, 4, in_channels, out_channels,
                                                4, 4, 'conv4x4', padding='SAME', act=tf.nn.relu)

        ## LAYER 2 ##

        # convolution layer for each 2x2 box representing each dinucleotide
        # in_channels = out_channels
        # out_channels1 = 16
        # layer2 = helpers.make_convolution_layer(layer1, 2, 2, in_channels, out_channels1,
        #                                           1, 1, 'conv2x2', padding='SAME', act=tf.nn.relu)

        # # make a parallel layer with for 6x6 box
        # out_channels2 = 16
        # layer2_2 = helpers.make_convolution_layer(layer1, 6, 6, in_channels, out_channels2,
        #                                           1, 1, 'layer2_2', padding='SAME', act=tf.nn.relu)

        # concatenate tensors:
        # layer2 = tf.concat(3, [layer2_1, layer2_2])

        # reshape to 1D tensor
        layer_flat_dim = layer1.get_shape().as_list()
        layer_flat_dim = layer_flat_dim[1] * layer_flat_dim[2] * layer_flat_dim[3]
        layer_flat = tf.reshape(layer1, [-1, layer_flat_dim])

        ## LAYER 3 ##

        # add fully connected layer
        in_channels = layer_flat_dim
        out_channels = 16
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
        out_channels = 1
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


        print('Reading in data')

        train, test = helpers.read_data(options.INFILE, LEN_DATA, DIM1, DIM2, extra=options.NUM_EXTRA, shuffle=True)
        print(len(train.labels))
        linmodel = LinearRegression()
        linmodel.fit(train.extra_features, train.labels)
        print(linmodel.score(test.extra_features, test.labels))


        print('Training model')

        # TRAIN MODEL #
        num_epoch = 40000
        batch_size = 50
        report_int = 1000
        keep_prob_train = 0.5

        # create a saver for saving the model after training
        saver = tf.train.Saver()

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # train epochs and record performance
        with open(options.ACCFILE, 'w') as acc_file:
            for i in range(1, num_epoch+1):
                batch = train.next_batch(batch_size)
                if i%report_int == 0:
                    
                    acc, summary = sess.run([accuracy, merged], feed_dict={x: test.features,
                                                                           extra: test.extra_features,
                                                                           keep_prob: 1.0,
                                                                           y: test.labels})
                    if options.SUMMARY:
                        test_writer.add_summary(summary, i)

                    print('Accuracy at step %s: %s' % (i, acc))


                    new_feat_train = sess.run(layer4, feed_dict={x: train.features,
                                                     extra: train.extra_features,
                                                     keep_prob: 1.0,
                                                     y: train.labels})

                    new_feat_test = sess.run(layer4, feed_dict={x: test.features,
                                                                extra: test.extra_features,
                                                                keep_prob: 1.0,
                                                                y: test.labels})

                    X_train = np.concatenate([train.extra_features, new_feat_train], axis=1)
                    X_test = np.concatenate([test.extra_features, new_feat_test], axis=1)

                    linmodel = LinearRegression()
                    linmodel.fit(X_train, train.labels)
                    print(linmodel.score(X_test, test.labels))

                    acc_file.write('{:.4}\n'.format(acc))

                    # save model
                    saver.save(sess, options.LOGDIR + '/my-model', global_step=i)
                
                else:
                    _, summary = sess.run([train_step, merged], feed_dict={x: batch[0],
                                                                           extra: batch[1],
                                                                           keep_prob: keep_prob_train,
                                                                           y: batch[2]})
                    if options.SUMMARY:
                        train_writer.add_summary(summary, i)

        new_feat_train = sess.run(layer4, feed_dict={x: train.features,
                                                     extra: train.extra_features,
                                                     keep_prob: 1.0,
                                                     y: train.labels})

        new_feat_test = sess.run(layer4, feed_dict={x: test.features,
                                                    extra: test.extra_features,
                                                    keep_prob: 1.0,
                                                    y: test.labels})

        X_train = np.concatenate([train.extra_features, new_feat_train], axis=1)
        X_test = np.concatenate([test.extra_features, new_feat_test], axis=1)

        linmodel = LinearRegression()
        linmodel.fit(X_train, train.labels)
        print(linmodel.score(X_test, test.labels))


