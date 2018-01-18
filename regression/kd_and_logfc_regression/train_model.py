import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tensorflow as tf

import helpers


def train_model(logfc_train_set, logfc_test_set, train_set, test_set, inference_func, params, logdir=None):

    if logdir is not None:
        if not os.path.isdir(logdir):
            os.makedirs(logdir)

    # reset and build the neural network
    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24))

    # if indicated, create directories to save the model
    if logdir is not None:
        SAVE_PATH = os.path.join(logdir, 'saved')
        VAR_PATH = os.path.join(logdir, 'vars')

        if not os.path.isdir(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        if not os.path.isdir(VAR_PATH):
            os.makedirs(VAR_PATH)

    # create placeholders for data
    x = tf.placeholder(tf.float32, shape=[None, 16 * params['MIRLEN'] * params['SEQLEN']], name='x')
    y = tf.placeholder(tf.float32, shape=[None, params['OUT_NODES']], name='y')
    x_image = tf.reshape(x, [-1, 4*params['MIRLEN'], 4*params['SEQLEN'], 1])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    num_kd = tf.placeholder(tf.int32, name='num_kd')
    num_logfc = tf.placeholder(tf.int32, name='num_logfc')

    # do inference step
    inference_results = inference_func(x_image, y, keep_prob, phase_train, num_kd, num_logfc, params)
    train_step, loss, prediction, var_dict = inference_results
    
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    var_saver = tf.train.Saver(var_dict)

    # get kd test data
    test_labels = test_set[['log_kd']].values
    test_features = np.zeros((len(test_set), 16 * params['MIRLEN'] * params['SEQLEN']))
    test_colors = list(test_set['color'])
    for i, (seq, mirseq) in enumerate(zip(test_set['seq'], test_set['mirseq'])):
        mirseq = mirseq[-params['MIRLEN']:]
        test_features[i,:] = helpers.make_square(mirseq, seq).flatten()
    
    # set up kd train data
    if params['TRAIN_KD']:
        train_set_temp = train_set.copy()
        train_set_temp['keep'] = [int(np.random.random() > 0.9) if x == 'grey' else 1 for x in train_set_temp['color']]
        train_set_temp = train_set_temp[train_set_temp['keep'] == 1]
    
    # get repression test data
    logfc_test_set_features = np.zeros((len(logfc_test_set), 16*params['MIRLEN']*params['SEQLEN']))
    logfc_test_set_colors = []
    logfc_test_set_labels = logfc_test_set[['logFC']].values

    for i, (mirseq, seq) in enumerate(zip(logfc_test_set['mirseq'], logfc_test_set['seq'])):
        mirseq = mirseq[-params['MIRLEN']:]
        logfc_test_set_features[i, :] = helpers.make_square(mirseq, seq).flatten()
        sitem8 = helpers.complementaryT(mirseq[-8:-1])
        logfc_test_set_colors.append(helpers.get_color_old(sitem8, seq))

    logfc_test_set['color'] = logfc_test_set_colors

    # set up repression train data
    if params['TRAIN_LOGFC']:
        logfc_train_set_temp = logfc_train_set.copy()
        logfc_train_set_temp['keep'] = [int(np.random.random() > 0.9) if x == 'grey' else 1 for x in logfc_train_set_temp['color']]
        logfc_train_set_temp = logfc_train_set_temp[logfc_train_set_temp['keep'] == 1]

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # if indicated, restore from previous model
    if params['RESTORE_FROM'] is not None:
        latest = tf.train.latest_checkpoint(params['RESTORE_FROM'])
        print('Restoring from {}'.format(latest))
        var_saver.restore(sess, latest)

    # train epochs and record performance
    sample_counter = 0
    batch_counter = 0
    epoch_counter = 0
    num_batches = []

    train_losses = []
    test_losses = []
    
    logfc_train_losses = []
    logfc_test_losses = []

    # must train on at least one dataset
    if not (params['TRAIN_KD'] or params['TRAIN_LOGFC']):
        raise ValueError('Either TRAIN_KD or TRAIN_LOGFC must be True.')

    while epoch_counter < params['MAX_EPOCH']:
        batch_train_features = []
        batch_train_labels = []
        batch_train_colors = []

        # train on kd data
        if params['TRAIN_KD']:

            # reset data if we finish an epoch
            if len(train_set_temp) < params['BATCH_SIZE']:
                train_set_temp = train_set.copy()
                train_set_temp['keep'] = [int(np.random.random() > 0.9) if x == 'grey' else 1 for x in train_set_temp['color']]
                train_set_temp = train_set_temp[train_set_temp['keep'] == 1]
                epoch_counter += 1
            
            # get batch
            subset = np.random.choice(list(train_set_temp.index), size=params['BATCH_SIZE'], replace=False)
            train_group = train_set_temp.loc[subset]
            train_set_temp = train_set_temp.drop(subset)

            batch_train_labels += list(train_group[['log_kd']].values)
            batch_train_colors += list(train_group['color'].values)

            for i, row in enumerate(train_group.iterrows()):
                mirseq = row[1]['mirseq'][-params['MIRLEN']:]
                batch_train_features.append(list(helpers.make_square(mirseq, row[1]['seq']).flatten()))

        # train on logfc data
        if params['TRAIN_LOGFC']:

            # reset data if we finish an epoch
            if len(logfc_train_set_temp) < params['LOGFC_BATCH_SIZE']:
                logfc_train_set_temp = logfc_train_set.copy()
                logfc_train_set_temp['keep'] = [int(np.random.random() > 0.9) if x == 'grey' else 1 for x in logfc_train_set_temp['color']]
                logfc_train_set_temp = logfc_train_set_temp[logfc_train_set_temp['keep'] == 1]

            # get batch
            subset = list(np.random.choice(list(logfc_train_set_temp.index), size=params['LOGFC_BATCH_SIZE'], replace=False))
            logfc_train_set_group = logfc_train_set_temp.loc[subset]
            logfc_train_set_temp = logfc_train_set_temp.drop(subset)

            batch_train_labels += list(logfc_train_set_group[['logFC']].values)
            batch_train_colors += list(logfc_train_set_group['color'].values)

            for i, row in enumerate(logfc_train_set_group.iterrows()):
                mirseq = row[1]['mirseq'][-params['MIRLEN']:]
                batch_train_features.append(list(helpers.make_square(mirseq, row[1]['seq']).flatten()))


        batch_train_features = np.array(batch_train_features)
        batch_train_labels = np.array(batch_train_labels)


        # train step
        _ = sess.run(train_step, feed_dict={x: batch_train_features,
                                            keep_prob: params['KEEP_PROB_TRAIN'],
                                            phase_train: True,
                                            num_kd: params['BATCH_SIZE'],
                                            num_logfc: params['LOGFC_BATCH_SIZE'],
                                            y: batch_train_labels})

        # increment batch counter
        batch_counter += 1
        
        # every few batches, record the loss
        if batch_counter%params['REPORT_INT'] == 0:

            test_loss, test_prediction = sess.run([loss, prediction], feed_dict={x: test_features,
                                                                                 keep_prob: 1.0,
                                                                                 phase_train: False,
                                                                                 num_kd: len(test_labels),
                                                                                 num_logfc: 0,
                                                                                 y: test_labels})


            logfc_test_loss, logfc_test_set_prediction = sess.run([loss, prediction],
                                                                   feed_dict={x: logfc_test_set_features,
                                                                               keep_prob: 1.0,
                                                                               phase_train: False,
                                                                               num_kd: 0,
                                                                               num_logfc: len(logfc_test_set_labels),
                                                                               y: logfc_test_set_labels
                                                                               })

            logfc_test_losses.append(logfc_test_loss)
            test_losses.append(test_loss)
            num_batches.append(batch_counter)

            # if indicated, plot the loss and performance and save the model
            if logdir is not None:

                # save the model
                saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=batch_counter)
                var_saver.save(sess, os.path.join(VAR_PATH, 'model'))

                # Plot kd loss over time
                fig = plt.figure()
                plt.plot(num_batches, test_losses)
                plt.xlabel('Num batches')
                plt.ylabel('Kd loss')
                plt.savefig(os.path.join(logdir, 'test_loss.pdf'))
                plt.close()

                # Plot repression loss over time
                fig = plt.figure()
                plt.plot(num_batches, logfc_test_losses)
                plt.xlabel('Num batches')
                plt.ylabel('Regression loss')
                plt.savefig(os.path.join(logdir, 'logfc_test_loss.pdf'))
                plt.close()

                # plot kd test predictions
                helpers.graph_predicted_v_actual(params['NCOLS'], params['OUT_NODES'], test_prediction, test_labels, 
                                                test_colors, os.path.join(logdir, 'scatter_test.pdf'),
                                                log_scale=params['LOG_SCALE'])

                # plot regression test predictions
                helpers.graph_predicted_v_actual(params['NCOLS'], params['OUT_NODES'], logfc_test_set_prediction,
                                                np.repeat(logfc_test_set_labels, params['OUT_NODES'], axis=1), 
                                                logfc_test_set_colors, os.path.join(logdir, 'scatter_test_logfc.pdf'),
                                                log_scale=params['LOG_SCALE'])

                
                train_loss, train_prediction = sess.run([loss, prediction], feed_dict={x: batch_train_features,
                                                                                       keep_prob: 1.0,
                                                                                       phase_train: False,
                                                                                       num_kd: params['BATCH_SIZE'],
                                                                                       num_logfc: params['LOGFC_BATCH_SIZE'],
                                                                                       y: batch_train_labels})

                helpers.graph_predicted_v_actual(params['NCOLS'], params['OUT_NODES'], train_prediction, batch_train_labels, 
                                                batch_train_colors, os.path.join(logdir, 'scatter_train.pdf'),
                                                log_scale=params['LOG_SCALE'])


                conv_weights = tf.get_collection('weight', scope='conv4x4')
                conv_weights = sess.run(conv_weights)[0]
                xlabels = ['U','A','G','C']
                ylabels = ['A','U','C','G']
                helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(logdir, 'convolution1.pdf'))

                # conv_weights = tf.get_collection('weight', scope='convlayer2')
                # conv_weights = sess.run(conv_weights)[0]
                # helpers.graph_convolutions(conv_weights, None, None, os.path.join(logdir, 'convolution2.pdf'))

                # conv_weights = tf.get_collection('weight', scope='convlayer3')
                # conv_weights = sess.run(conv_weights)[0]
                # helpers.graph_convolutions(conv_weights, None, None, os.path.join(logdir, 'convolution3.pdf'))

                # conv_weights = tf.get_collection('weight', scope='fullyconnected')
                # conv_weights = sess.run(conv_weights)[0]
                # dim = conv_weights.shape
                # fig = plt.figure(figsize=(dim[1], dim[0]))

                # sns.heatmap(conv_weights, cmap=plt.cm.bwr)
                # plt.savefig(os.path.join(logdir, 'convolution_fullyconnected.pdf'))
                # plt.close()

            # do early stopping if test loss stops going down
            # if len(logfc_test_losses) >= 20:
            #     if np.mean(logfc_test_losses[-10:]) >= np.mean(logfc_test_losses[-20:]):
            #         sess.close()
            #         return test_losses, logfc_test_losses, epoch_counter

    sess.close()
    return test_losses, logfc_test_losses, epoch_counter
