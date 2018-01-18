import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf

import helpers


def train_model(train_set, test_set, logfc_test_set, adv_data, inference_func, params, logdir=None):

    # if indicated, create directories to save the model and graph outputs
    if logdir is not None:
        if not os.path.isdir(logdir):
            os.makedirs(logdir)

        SAVE_PATH = os.path.join(logdir, 'saved')
        VAR_PATH = os.path.join(logdir, 'vars')

        if not os.path.isdir(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        if not os.path.isdir(VAR_PATH):
            os.makedirs(VAR_PATH)

    # reset and build the neural network
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:
    # with tf.Session() as sess:

        # create placeholders for data
        sequence_encoding = tf.placeholder(tf.float32, shape=[None, 16 * params['MIRLEN'] * params['SEQLEN']], name='sequence_encoding')
        labels = tf.placeholder(tf.float32, shape=[None, params['OUT_NODES']], name='label')
        mask = tf.placeholder(tf.float32, shape=[None, 1], name='mask')
        mask2 = tf.placeholder(tf.float32, shape=[None], name='mask2')
        classes = tf.placeholder(tf.int32, shape=[None, params['NUM_CLASSES']], name='class')
        sequence_encoding_2D = tf.reshape(sequence_encoding, [-1, 4*params['MIRLEN'], 4*params['SEQLEN'], 1])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        # do inference step
        model_results = inference_func(sequence_encoding_2D, keep_prob, phase_train, params)
        encoding, prediction, discriminator, var_dict, all_vars, D_vars = model_results

        # calculate losses
        with tf.name_scope('loss'):
            # loss_disc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=classes, logits=discriminator))
            loss_disc_pre = tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=classes, logits=discriminator), mask2)
            loss_disc = tf.reduce_sum(loss_disc_pre) / tf.reduce_sum(mask2)
            correct_prediction = tf.equal(tf.argmax(classes, 1), tf.argmax(discriminator, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            loss_l2 = tf.reduce_mean(tf.multiply(tf.subtract(prediction, labels)**2, mask))
            loss = loss_l2 - tf.multiply(loss_disc, params['LAMBDA'])

        # get train step
        with tf.name_scope('train_all'):

            # required for updating batch_norm variables
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                optimizer = tf.train.AdamOptimizer(params['STARTING_LEARNING_RATE_L2'])
                all_grads = tf.gradients(loss, all_vars)
                train_step_all = optimizer.apply_gradients(zip(all_grads, all_vars))

        # discriminator train step
        with tf.name_scope('train_D'):
            D_optimizer = tf.train.AdamOptimizer(params['STARTING_LEARNING_RATE_DISC'])
            D_grads = tf.gradients(loss_disc, D_vars)
            train_step_disc = D_optimizer.apply_gradients(zip(D_grads, D_vars))
        
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
        train_set_temp = train_set.copy()
        train_set_temp['keep'] = [int(np.random.random() > 0.9) if x == 'grey' else 1 for x in train_set_temp['color']]
        train_set_temp = train_set_temp[train_set_temp['keep'] == 1]
        
        # get repression test data
        logfc_test_set_features = np.zeros((len(logfc_test_set), 16*params['MIRLEN']*params['SEQLEN']))
        logfc_test_set_colors = list(logfc_test_set['color'].values)
        logfc_test_set_labels = logfc_test_set[['logFC']].values

        for i, (mirseq, seq) in enumerate(zip(logfc_test_set['mirseq'], logfc_test_set['seq'])):
            mirseq = mirseq[-params['MIRLEN']:]
            logfc_test_set_features[i, :] = helpers.make_square(mirseq, seq).flatten()

        # set up data for adversarial training
        adv_data_temp = adv_data.copy()

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

        # class_entropy = 5*(-1*0.2*np.log(0.2))
        p1 = params['BATCH_SIZE']/(params['BATCH_SIZE'] + params['ADV_BATCH_SIZE'])
        p2 = 1.0 - p1
        class_entropy = (-1*p1)*np.log(p1) + (-1*p2)*np.log(p2)
        print(class_entropy)

        while epoch_counter < params['MAX_EPOCH']:

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

            batch_train_labels = np.concatenate([train_group[['log_kd']].values, np.array([[0.0]]*params['ADV_BATCH_SIZE'])], axis=0)
            batch_train_colors = list(train_group['color'].values)
            batch_train_classes = np.array([[1,0]]*params['BATCH_SIZE'] + [[0,1]]*params['ADV_BATCH_SIZE'])
            batch_train_mask = np.array([[1]]*params['BATCH_SIZE'] + [[0]]*params['ADV_BATCH_SIZE'])
            batch_train_mask2 = []

            batch_train_features = []
            for row in train_group.iterrows():
                mirseq = row[1]['mirseq'][-params['MIRLEN']:]
                batch_train_features.append(list(helpers.make_square(mirseq, row[1]['seq']).flatten()))
                if row[1]['color'] == 'grey':
                    batch_train_mask2.append(0)
                else:
                    batch_train_mask2.append(1)

            # reset adv data
            if len(adv_data_temp) < params['BATCH_SIZE']:
                adv_data_temp = adv_data.copy()

            subset = np.random.choice(list(adv_data_temp.index), size=params['ADV_BATCH_SIZE'], replace=False)
            adv_batch = adv_data_temp.loc[subset]
            adv_data_temp = adv_data_temp.drop(subset)

            adv_batch_colors = list(adv_batch['color'].values)

            for row in adv_batch.iterrows():
                mirseq = row[1]['mirseq'][-params['MIRLEN']:]
                batch_train_features.append(list(helpers.make_square(mirseq, row[1]['seq']).flatten()))
                if row[1]['color'] == 'grey':
                    batch_train_mask2.append(0)
                else:
                    batch_train_mask2.append(1)

            batch_train_features = np.array(batch_train_features)
            batch_train_mask2 = np.array(batch_train_mask2)

            # train step
            _, current_grads  = sess.run([train_step_all, all_grads],
                            feed_dict={
                                    sequence_encoding: batch_train_features,
                                    classes: batch_train_classes,
                                    mask: batch_train_mask,
                                    mask2: batch_train_mask2,
                                    keep_prob: params['KEEP_PROB_TRAIN'],
                                    phase_train: True,
                                    labels: batch_train_labels
                            }
                )

            sess.run(tf.variables_initializer(D_vars))

            ix = 0
            current_loss_disc = 1
            while (current_loss_disc > class_entropy):
                _, current_loss_disc, acc = sess.run([train_step_disc, loss_disc, accuracy],
                                                feed_dict={
                                                            sequence_encoding: batch_train_features,
                                                            keep_prob: 1.0,
                                                            phase_train: False,
                                                            classes: batch_train_classes,
                                                            mask: batch_train_mask,
                                                            mask2: batch_train_mask2,
                                                            labels: batch_train_labels
                                                }
                    )
                ix += 1
                # print(current_loss_disc)
                if ix > 50:
                    print('BLARGG')
                    return np.nan, np.nan, epoch_counter

            # print(ix, current_loss_disc)
            
            # every few batches, record the loss
            if batch_counter%params['REPORT_INT'] == 0:

                acc, current_loss_disc = sess.run([accuracy, loss_disc],
                                            feed_dict={
                                                    sequence_encoding: batch_train_features,
                                                    classes: batch_train_classes,
                                                    mask: batch_train_mask,
                                                    mask2: batch_train_mask2,
                                                    keep_prob: 1.0,
                                                    phase_train: False
                                            }
                )
                # print('**********')
                # print(acc, current_loss_disc)
                # print('**********')

                test_loss, test_prediction = sess.run([loss_l2, prediction], feed_dict={sequence_encoding: test_features,
                                                                                     keep_prob: 1.0,
                                                                                     phase_train: False,
                                                                                     mask: np.array([[1]]*len(test_features)),
                                                                                     mask2: np.array([1]*len(test_features)),
                                                                                     labels: test_labels})


                logfc_test_loss, logfc_test_set_prediction = sess.run([loss_l2, prediction],
                                                                       feed_dict={sequence_encoding: logfc_test_set_features,
                                                                                   keep_prob: 1.0,
                                                                                   phase_train: False,
                                                                                   mask: np.array([[1]]*len(logfc_test_set_features)),
                                                                                   mask2: np.array([1]*len(logfc_test_set_features)),
                                                                                   labels: logfc_test_set_labels
                                                                                   })

                logfc_test_losses.append(logfc_test_loss)
                test_losses.append(test_loss)
                num_batches.append(batch_counter)

                # print(logfc_test_loss, test_loss)
                if np.isnan(logfc_test_loss) or np.isnan(test_loss):
                    print(logfc_test_loss)
                    print(test_loss)
                    return np.nan, np.nan, epoch_counter

                # if indicated, plot the loss and performance and save the model
                if logdir is not None:

                    # save the model
                    saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=batch_counter)
                    var_saver.save(sess, os.path.join(VAR_PATH, 'model'))

                    # plot pca of encoding
                    batch_encoding = sess.run(encoding, feed_dict={sequence_encoding: batch_train_features, keep_prob: 1.0, phase_train: False})
                    not_grey = np.argwhere(batch_train_mask2).flatten()
                    ix_train = [x for x in not_grey if x < params['BATCH_SIZE']]
                    ix_test = [x for x in not_grey if x >= params['BATCH_SIZE']]
                    fig = plt.figure()
                    pca = TSNE(n_components=2)
                    transformed = pca.fit_transform(batch_encoding)
                    xs,ys = transformed[ix_train,0], transformed[ix_train,1]
                    plt.scatter(xs,ys, color=[x for x in batch_train_colors if x != 'grey'], label='sean_mirs')
                    xs,ys = transformed[ix_test,0], transformed[ix_test,1]
                    plt.scatter(xs,ys, facecolors='none', edgecolors=[x for x in adv_batch_colors if x != 'grey'], label='other_mirs')
                    plt.legend()
                    plt.title('{:.3}'.format(acc))
                    plt.savefig(os.path.join(logdir, 'encoding_pca.pdf'))
                    plt.close()

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

                    # plot repression test predictions
                    logfc_test_set['prediction'] = logfc_test_set_prediction
                    temp = logfc_test_set[logfc_test_set['mir'].isin(['mir1','mir124','mir155'])]
                    preds = temp[['prediction']].values
                    labs = temp[['logFC']].values
                    cols = list(temp['color'].values)
                    helpers.graph_predicted_v_actual(params['NCOLS'], params['OUT_NODES'], preds,
                                                    labs, 
                                                    cols, os.path.join(logdir, 'scatter_test_logfc_train.pdf'),
                                                    log_scale=params['LOG_SCALE'])

                    temp = logfc_test_set[~logfc_test_set['mir'].isin(['mir1','mir124','mir155'])]
                    preds = temp[['prediction']].values
                    labs = temp[['logFC']].values
                    cols = list(temp['color'].values)
                    helpers.graph_predicted_v_actual(params['NCOLS'], params['OUT_NODES'], preds,
                                                    labs, 
                                                    cols, os.path.join(logdir, 'scatter_test_logfc_test.pdf'),
                                                    log_scale=params['LOG_SCALE'])

                    
                    train_prediction = sess.run(prediction, feed_dict={sequence_encoding: batch_train_features,
                                                                       keep_prob: 1.0,
                                                                       phase_train: False})

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
                # if len(logfc_test_losses) >= 11:
                #     if np.median(logfc_test_losses[-5:]) >= np.median(logfc_test_losses[-11:]):
                #         return test_loss, logfc_test_loss, epoch_counter

            # increment batch counter
            batch_counter += 1

        return test_loss, logfc_test_loss, epoch_counter
