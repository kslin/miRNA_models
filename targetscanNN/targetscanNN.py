import sys

import numpy as np
import tensorflow as tf

import helpers


if __name__ == '__main__':

    INFILE, LEN_DATA, LEN_FEATURES, LOGDIR = sys.argv[1:]
    LEN_DATA, LEN_FEATURES = int(LEN_DATA), int(LEN_FEATURES)
    WRITE_SUMMARY = False

    print('Building neural network')

    with tf.Session() as sess:

        dim1, dim2 = 80, 96
        num_extra = 17
        out_nodes = 1

        # create placeholders for data
        x = tf.placeholder(tf.float32, shape=[None, dim1*dim2])
        extra = tf.placeholder(tf.float32, shape=[None, num_extra])
        y = tf.placeholder(tf.float32, shape=[None, out_nodes])
        x_image = tf.reshape(x, [-1, dim1, dim2, 1])
        
        # create summaries of features for visualizing with tensorboard
        x_summary = tf.summary.image('images', x_image)

        ## LAYER 1 ##

        # convolution layer for each 4x4 box representing two paired nucleotides
        in_channels = 1
        out_channels = 16
        layer1 = helpers.make_convolution_layer(x_image, 4, 4, in_channels, out_channels,
                                                4, 4, 'layer1', padding='SAME', act=tf.nn.relu)

        ## LAYER 2 ##

        # convolution layer for each 2x2 box representing each dinucleotide
        in_channels = out_channels
        out_channels1 = 16
        layer2 = helpers.make_convolution_layer(layer1, 2, 2, in_channels, out_channels1,
                                                  1, 1, 'layer2_1', padding='SAME', act=tf.nn.relu)

        # # make a parallel layer with for 6x6 box
        # out_channels2 = 16
        # layer2_2 = helpers.make_convolution_layer(layer1, 6, 6, in_channels, out_channels2,
        #                                           1, 1, 'layer2_2', padding='SAME', act=tf.nn.relu)

        # concatenate tensors:
        # layer2 = tf.concat(3, [layer2_1, layer2_2])

        # reshape to 1D tensor
        layer2_flat_dim = layer2.get_shape().as_list()
        layer2_flat_dim = layer2_flat_dim[1] * layer2_flat_dim[2] * layer2_flat_dim[3]
        layer2_flat = tf.reshape(layer2, [-1, layer2_flat_dim])

        # add in extra features
        # layer2_flat = tf.concat(1, [tf.reshape(layer2, [-1, layer2_flat_dim]), extra])

        ## LAYER 3 ##

        # add fully connected layer
        in_channels = layer2_flat_dim# + num_extra
        out_channels = 8
        layer4 = helpers.make_fullyconnected_layer(layer2_flat, in_channels, out_channels, 'fullyconnected', act=tf.nn.relu)

        # # add dropout
        # with tf.name_scope('dropout'):
        #     keep_prob = tf.placeholder(tf.float32)
        #     tf.summary.scalar('dropout_keep_probability', keep_prob)
        #     dropout = tf.nn.dropout(layer3, keep_prob)

        ## LAYER 4 ##

        # add last layer with 1 output channel
        # in_channels = out_channels
        # out_channels = 1
        # layer4 = helpers.make_fullyconnected_layer(dropout, in_channels, out_channels, 'fullyconnected2', act=tf.identity)

        # add extra features
        layer4 = tf.concat(1, [layer4, extra])


        ## LAYER 5 ##

        # add last layer with 1 output channel
        in_channels = out_channels + num_extra
        out_channels = out_nodes
        layer5 = helpers.make_fullyconnected_layer(layer4, in_channels, out_channels, 'fullyconnected3', act=tf.identity)


        # prepare training steps and log writers
        train_step, accuracy = helpers.make_train_step('regression', layer5, y)

        merged = tf.summary.merge_all()

        if WRITE_SUMMARY:
            train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(LOGDIR + '/test')


        print('Reading in data')

        train, test = helpers.read_data(INFILE, LEN_DATA, LEN_FEATURES, extra=num_extra, shuffle=True)

        print('Training model')

        # TRAIN MODEL #
        num_epoch = 20000
        batch_size = 100
        report_int = 500
        keep_prob_train = 0.5

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # train epochs
        for i in range(num_epoch):
            batch = train.next_batch(batch_size)
            if i%report_int == 0:
                
                acc, summary = sess.run([accuracy, merged], feed_dict={x: test.features,
                                                                       extra: test.extra_features,
                                                                       # keep_prob: 1.0
                                                                       y: test.labels})
                if WRITE_SUMMARY:
                    test_writer.add_summary(summary, i)

                print('Accuracy at step %s: %s' % (i, acc))
            
            else:
                _, summary = sess.run([train_step, merged], feed_dict={x: batch[0],
                                                                       extra: batch[1],
                                                                       # keep_prob: keep_prob_train,
                                                                       y: batch[2]})
                if WRITE_SUMMARY:
                    train_writer.add_summary(summary, i)

        print("test accuracy %g"%accuracy.eval(feed_dict={x: test.features,
                                                          extra: test.extra_features,
                                                          # keep_prob: 1.0,
                                                          y: test.labels}))

