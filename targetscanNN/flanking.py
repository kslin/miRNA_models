import sys

import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf

import helpers


if __name__ == '__main__':

    INFILE, LEN_DATA, LOGDIR = sys.argv[1:]
    LEN_DATA = int(LEN_DATA)
    WRITE_SUMMARY = False

    print('Building neural network')

    with tf.Session() as sess:
        dim = 28*4
        out_nodes = 1

        # create placeholders for data
        x = tf.placeholder(tf.float32, shape=[None, dim])
        y = tf.placeholder(tf.float32, shape=[None, out_nodes])

        # make a fully connected layer
        in_channels = dim
        out_channels = 1024
        layer1 = helpers.make_fullyconnected_layer(x, in_channels, out_channels, 'fullyconnected1', act=tf.nn.relu)


        # add dropout
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropout = tf.nn.dropout(layer1, keep_prob)


        # get output
        in_channels = out_channels
        out_channels = out_nodes
        layer1 = helpers.make_fullyconnected_layer(dropout, in_channels, out_channels, 'fullyconnected2', act=tf.identity)

        # prepare training steps and log writers
        train_step, accuracy = helpers.make_train_step('regression', layer1, y)

        merged = tf.summary.merge_all()

        if WRITE_SUMMARY:
            train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(LOGDIR + '/test')


        print('Reading in data')

        train, test = helpers.read_flanking(INFILE, LEN_DATA, dim, shuffle=True)
        # print(len(train.labels))
        # linmodel = LinearRegression()
        # linmodel.fit(train.extra_features, train.labels)
        # print(linmodel.score(test.extra_features, test.labels))

        print('Training model')

        # TRAIN MODEL #
        num_epoch = 30000
        batch_size = 50
        report_int = 1000
        keep_prob_train = 0.5

        # create a saver for saving the model after training
        saver = tf.train.Saver()

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # train epochs
        for i in range(1, num_epoch+1):
            batch = train.next_batch(batch_size)
            if i%report_int == 0:
                
                acc, summary = sess.run([accuracy, merged], feed_dict={x: test.features,
                                                                       # extra: test.extra_features,
                                                                       keep_prob: 1.0,
                                                                       y: test.labels})
                if WRITE_SUMMARY:
                    test_writer.add_summary(summary, i)

                print('Accuracy at step %s: %s' % (i, acc))

                # save model
                saver.save(sess, LOGDIR + '/my-model', global_step=i)
            
            else:
                _, summary = sess.run([train_step, merged], feed_dict={x: batch[0],
                                                                       # extra: batch[1],
                                                                       keep_prob: keep_prob_train,
                                                                       y: batch[2]})
                if WRITE_SUMMARY:
                    train_writer.add_summary(summary, i)
