import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

import tensorflow as tf

import config
import helpers
import model


IN_NODES = 1
OUT_NODES = 1
NCOLS = np.ceil(OUT_NODES/2)
HIDDEN1 = 4
HIDDEN2 = 2
HIDDEN3 = 16
STARTING_LEARNING_RATE = 1e-3
ERROR_MODEL = 'poisson'
LOG_SCALE = (ERROR_MODEL == 'poisson')

NUM_EPOCH = 10
BATCH_SIZE = 200
REPORT_INT = 20
KEEP_PROB_TRAIN = 0.5
TEST_SIZE = 5000

# RESTORE_FROM = '/lab/bartel4_ata/kathyl/NeuralNet/logdirs/simple_match/vars'
RESTORE_FROM = None
LOGDIR = '/lab/bartel4_ata/kathyl/NeuralNet/logdirs/simple_match_4_2_16'

if not os.path.isdir(LOGDIR):
    os.makedirs(LOGDIR)

SAVE_PATH = os.path.join(LOGDIR, 'saved')

if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)

VAR_PATH = os.path.join(LOGDIR, 'vars')

if not os.path.isdir(VAR_PATH):
    os.makedirs(VAR_PATH)

def generate_random_pairs():
    # while True:
    #     mirseq = helpers.generate_random_seq(20)
    #     sitem8 = helpers.complementaryT(mirseq[-8:-1])
    #     seq = helpers.generate_random_seq(8) + sitem8[1:] + helpers.generate_random_seq(8)
    #     r = np.random.randint(10)
    #     seq = seq[r:r+12]
    #     color = helpers.get_color(sitem8, seq)

    #     if color != 'offcenter':
    #         break

    mirseq = helpers.generate_random_seq(20)
    sitem8 = helpers.complementaryT(mirseq[-8:-1])
    seq = helpers.generate_random_seq(8) + sitem8[1:] + helpers.generate_random_seq(8)
    r = np.random.randint(10)
    seq = seq[r:r+12]
    color = helpers.get_color_old(sitem8, seq)

    value = 1
    if color == 'blue':
        value = 1000
    elif color == 'green':
        value = 550
    elif color == 'orange':
        value = 450
    elif color == 'red':
        value = 100

    return mirseq, seq, color, stats.poisson.rvs(value)

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:

    print('Creating graph')

    # create placeholders for data
    x = tf.placeholder(tf.float32, shape=[None, 16 * config.MIRLEN * config.SEQLEN])
    y = tf.placeholder(tf.float32, shape=[None, OUT_NODES])
    x_image = tf.reshape(x, [-1, 4*config.MIRLEN, 4*config.SEQLEN, 1])

    keep_prob = tf.placeholder(tf.float32)

    train_step, accuracy, loss, prediction, var_dict = model.inference(x_image, y, IN_NODES, HIDDEN1, HIDDEN2, HIDDEN3,
                                                              keep_prob, OUT_NODES, STARTING_LEARNING_RATE, ERROR_MODEL)

    merged = tf.summary.merge_all()

    print(var_dict)

    var_saver = tf.train.Saver(var_dict)
    # var_saver = tf.train.Saver({x:y for (x,y) in var_dict.items() if x in ['conv4x4_weight', "conv4x4_bias"]})

    saver = tf.train.Saver()

    # create test set:
    test_features = np.zeros((TEST_SIZE, 16 * config.MIRLEN * config.SEQLEN))
    test_labels = np.zeros((TEST_SIZE, OUT_NODES))
    test_colors = np.array([None]*TEST_SIZE)

    for i in range(TEST_SIZE):
        mirseq, seq, color, value = generate_random_pairs()
        test_features[i, :] = helpers.make_square(mirseq, seq).flatten()
        test_labels[i, :] = [value]
        test_colors[i] = color


    print('Training model')

    sess.run(tf.global_variables_initializer())

    if RESTORE_FROM is not None:
        latest = tf.train.latest_checkpoint(RESTORE_FROM)
        print(latest)
        var_saver.restore(sess, latest)

    # train epochs and record performance
    sample_counter = 0
    batch_counter = 1
    epoch_counter = 0
    train_losses = []
    test_losses = []
    test_steps = []

    train_features = np.zeros((BATCH_SIZE, 16 * config.MIRLEN * config.SEQLEN))
    train_labels = np.zeros((BATCH_SIZE, OUT_NODES))
    train_colors = np.array([None]*BATCH_SIZE)

    while True:

        # generate random data point
        mirseq, seq, color, value = generate_random_pairs()
        train_colors[sample_counter] = color
        train_features[sample_counter, :] = helpers.make_square(mirseq, seq).flatten()
        train_labels[sample_counter, :] = [value]
        sample_counter += 1

        # do train step once we collect a minibatch
        if sample_counter == BATCH_SIZE:
            batch_counter += 1
            sample_counter = 0

            if batch_counter%REPORT_INT != 0:
                _ = sess.run(train_step, feed_dict={
                        x: train_features,
                        keep_prob: KEEP_PROB_TRAIN,
                        y: train_labels})
            
            else:

                _, train_loss, train_prediction = sess.run([train_step, loss, prediction],
                                                            feed_dict={x: train_features,
                                                                       keep_prob: KEEP_PROB_TRAIN,
                                                                       y: train_labels})

                test_loss, test_prediction = sess.run([loss, prediction], feed_dict={x: test_features,
                                                                                     keep_prob: 1.0,
                                                                                     y: test_labels})

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                test_steps.append(batch_counter)

                # print progress
                print('Step %s, train loss: %s, test loss: %s' % (batch_counter, train_loss, test_loss))

                # save model
                saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=batch_counter)
                var_saver.save(sess, os.path.join(VAR_PATH, 'vars'), global_step=batch_counter)

                # Plot loss over time
                fig = plt.figure()
                plt.plot(test_steps, train_losses)
                plt.savefig(os.path.join(LOGDIR, 'train_loss.pdf'))
                plt.close()

                fig = plt.figure()
                plt.plot(test_steps, test_losses)
                plt.savefig(os.path.join(LOGDIR, 'test_loss.pdf'))
                plt.close()

                helpers.graph_predicted_v_actual(NCOLS, OUT_NODES, train_prediction, train_labels, 
                                                train_colors, os.path.join(LOGDIR, 'scatter_train.pdf'),
                                                log_scale=LOG_SCALE)

                helpers.graph_predicted_v_actual(NCOLS, OUT_NODES, test_prediction, test_labels, 
                                                test_colors, os.path.join(LOGDIR, 'scatter_test.pdf'),
                                                log_scale=LOG_SCALE)

                conv_weights = tf.get_collection('weight', scope='conv4x4')
                conv_weights = sess.run(conv_weights)[0]
                xlabels = ['U','A','G','C']
                ylabels = ['A','U','C','G']
                helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(LOGDIR, 'convolution1.pdf'))

                conv_weights = tf.get_collection('weight', scope='convlayer2')
                conv_weights = sess.run(conv_weights)[0]
                helpers.graph_convolutions(conv_weights, None, None, os.path.join(LOGDIR, 'convolution2.pdf'))

                # conv_weights = tf.get_collection('weight', scope='convlayer3')
                # conv_weights = sess.run(conv_weights)[0]
                # helpers.graph_convolutions(conv_weights, None, None, os.path.join(LOGDIR, 'convolution3.pdf'))



