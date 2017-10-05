import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import os
import pandas as pd
from scipy import stats

import tensorflow as tf

import config
import helpers
import model

LOG_SCALE = (config.params['ERROR_MODEL'] == 'poisson')
NCOLS = np.ceil(config.params['OUT_NODES']/2)

def train_model(logdir, infile, metafile):

    SAVE_PATH = os.path.join(logdir, 'saved')

    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    VAR_PATH = os.path.join(logdir, 'vars')

    if not os.path.isdir(VAR_PATH):
        os.makedirs(VAR_PATH)

    for key in sorted(config.params.keys()):
        metafile.write('{}: {}\n'.format(key, config.params[key]))

    # build neural network
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:

        print('Creating graph')

        # create placeholders for data
        x = tf.placeholder(tf.float32, shape=[None, 16 * config.params['MIRLEN'] * config.params['SEQLEN']], name='x')
        y = tf.placeholder(tf.float32, shape=[None, config.params['OUT_NODES']], name='y')
        if config.params['BASELINE_NODES'] > 0:
            baseline = tf.placeholder(tf.float32, shape=[None, config.params['BASELINE_NODES']], name='baseline')
        else:
            baseline = tf.placeholder(tf.float32, shape=[None, 1], name='baseline')
        x_image = tf.reshape(x, [-1, 4*config.params['MIRLEN'], 4*config.params['SEQLEN'], 1])

        data_weights = tf.placeholder(tf.float32, shape=[None, 1], name='data_weights')

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        train_step, accuracy, loss, prediction, var_dict = model.inference(x_image, y, baseline,
                                                                           config.params['IN_NODES'], config.params['OUT_NODES'],
                                                                           config.params['BASELINE_NODES'],
                                                                           config.params['HIDDEN1'], config.params['HIDDEN2'],
                                                                           config.params['HIDDEN3'], keep_prob, 
                                                                           config.params['STARTING_LEARNING_RATE'],
                                                                           config.params['ERROR_MODEL'], logdir, data_weights)

        merged = tf.summary.merge_all()

        saver = tf.train.Saver()
        var_saver = tf.train.Saver(var_dict)

        # read in log fold-change data
        print('Reading logFC data')

        logfc = pd.read_csv('/lab/bartel4_ata/kathyl/NeuralNet/data/logfc/12mers_agg.txt',sep='\t')
        logfc_features = np.zeros((len(logfc), 16*config.params['MIRLEN']*config.params['SEQLEN']))
        logfc_colors = []
        logfc_labels = logfc[['logFC']].values

        for i, (mirseq, seq) in enumerate(zip(logfc['miRNA_sequence'], logfc['flanking'])):
            mirseq = mirseq[-config.params['MIRLEN']:]
            logfc_features[i, :] = helpers.make_square(mirseq, seq).flatten()
            sitem8 = helpers.complementaryT(mirseq[-8:-1])
            logfc_colors.append(helpers.get_color_old(sitem8, seq))

        logfc['color'] = logfc_colors

        # read in test set:
        print('Reading test data')
        data = pd.read_csv(infile, header=None, nrows=config.params['TEST_SIZE'])
        data.columns = ['mirseq','seq'] + list(range(config.params['BASELINE_NODES'] + config.params['OUT_NODES']))
        data['sitem8'] = [helpers.complementaryT(mirseq[-8:-1]) for mirseq in data['mirseq']]
        data['color'] = [helpers.get_color_old(sitem8, seq) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

        # data = data[data['color'] != 'offcenter']

        print("Length of test data: {}".format(len(data)))

        if config.params['KD']:
            test_labels = np.log2(data[list(range(config.params['BASELINE_NODES'], config.params['BASELINE_NODES']+config.params['OUT_NODES']))].values.astype(float))
        else:
            test_labels = np.log2(data[list(range(config.params['BASELINE_NODES'], config.params['BASELINE_NODES']+config.params['OUT_NODES']))].values.astype(float) + 1)

        print(test_labels[:10, :])

        if config.params['BASELINE_NODES'] > 0:
            test_baseline = data[list(range(config.params['BASELINE_NODES']))].values.astype(float)
            test_baseline = np.log2(test_baseline.reshape((len(data), config.params['BASELINE_NODES'])))

            mean_baseline = np.mean(test_baseline, axis=0).reshape(1,config.params['BASELINE_NODES'])
            logfc_baseline = np.repeat(mean_baseline, len(logfc), axis=0)
            print(mean_baseline)

        else:
            test_baseline = np.zeros((len(data), 1))
            logfc_baseline = np.zeros((len(logfc), 1))


        

        # test_labels = data[[4]].values.astype(float)
        

        test_features = np.zeros((len(data), 16 * config.params['MIRLEN'] * config.params['SEQLEN']))
        test_colors = list(data['color'])
        for i, (seq, mirseq) in enumerate(zip(data['seq'], data['mirseq'])):
            mirseq = mirseq[-config.params['MIRLEN']:]
            test_features[i,:] = helpers.make_square(mirseq, seq).flatten()


        print('Training model')

        sess.run(tf.global_variables_initializer())

        if config.params['RESTORE_FROM'] is not None:
            latest = tf.train.latest_checkpoint(config.params['RESTORE_FROM'])
            print(latest)
            var_saver.restore(sess, latest)

        # train epochs and record performance
        sample_counter = 0
        batch_counter = 1
        epoch_counter = 0
        train_losses = []
        test_losses = []
        test_steps = []

        train_features = np.zeros((config.params['BATCH_SIZE'], 16 * config.params['MIRLEN'] * config.params['SEQLEN']))
        if config.params['BASELINE_NODES'] > 0:
            train_baseline = np.zeros((config.params['BATCH_SIZE'], config.params['BASELINE_NODES']))
        else:
            train_baseline = np.zeros((config.params['BATCH_SIZE'], 1))

        train_labels = np.zeros((config.params['BATCH_SIZE'], config.params['OUT_NODES']))
        train_colors = np.array([None]*config.params['BATCH_SIZE'])

        reader = open(infile, 'r')
        skip_size = 0
        for i in range(config.params['TEST_SIZE']):
            line = reader.readline()
            skip_size += len(line)

        while True:

            # collect next data point
            line = reader.readline()
            if len(line) == 0:
                reader.seek(skip_size)
                line = reader.readline()
                epoch_counter += 1
                metafile.write('finished epoch {}\n'.format(epoch_counter))
                if epoch_counter == config.params['NUM_EPOCH']:
                    break


            vals = line.replace('\n','').split(',')
            mirseq = vals[0][-config.params['MIRLEN']:]
            seq = vals[1]
            sitem8 = helpers.complementaryT(mirseq[-8:-1])
            color = helpers.get_color_old(sitem8, seq)
            if color == 'grey':
                if np.random.random() > 0.05:
                    continue

            train_colors[sample_counter] = color
            train_features[sample_counter, :] = helpers.make_square(mirseq, seq).flatten()
            
            if config.params['KD']:
                train_labels[sample_counter, :] = np.log2(np.array(vals[2 + config.params['BASELINE_NODES']:]).astype(float))
            else:
                train_labels[sample_counter, :] = np.log2(np.array(vals[2 + config.params['BASELINE_NODES']:]).astype(float) + 1)

            if config.params['BASELINE_NODES'] > 0:
                train_baseline[sample_counter, :] = np.log2(np.array(vals[2:2 + config.params['BASELINE_NODES']]).astype(float) + 1)

            sample_counter += 1

            # do train step once we collect a minibatch
            if sample_counter == config.params['BATCH_SIZE']:
                batch_counter += 1
                sample_counter = 0

                if batch_counter%config.params['REPORT_INT'] != 0:
                    _ = sess.run(train_step, feed_dict={
                            x: train_features,
                            baseline: train_baseline,
                            keep_prob: config.params['KEEP_PROB_TRAIN'],
                            data_weights: np.ones((config.params['BATCH_SIZE'], 1)),
                            y: train_labels})
                
                else:

                    _, train_loss, train_prediction = sess.run([train_step, loss, prediction],
                                                                feed_dict={x: train_features,
                                                                           baseline: train_baseline,
                                                                           keep_prob: config.params['KEEP_PROB_TRAIN'],
                                                                           data_weights: np.ones((config.params['BATCH_SIZE'], 1)),
                                                                           y: train_labels})

                    test_loss, test_prediction = sess.run([loss, prediction], feed_dict={x: test_features,
                                                                                         baseline: test_baseline,
                                                                                         keep_prob: 1.0,
                                                                                         data_weights: np.ones((config.params['TEST_SIZE'], 1)),
                                                                                         y: test_labels})

                    logfc_prediction = sess.run(prediction, feed_dict={x: logfc_features,
                                                                       baseline: logfc_baseline,
                                                                       keep_prob: 1.0
                                                                       })

                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    test_steps.append(batch_counter)

                    # do early stopping if test loss stops going down
                    if len(test_losses) >= 100:
                        if np.mean(test_losses[-50:]) >= np.mean(test_losses[-100:]):
                            metafile.write("\nStopping early at epoch {}".format(epoch_counter + 1))
                            break

                    # print progress
                    # print('Step %s, train loss: %s, test loss: %s' % (batch_counter, train_loss, test_loss))

                    # save model
                    saver.save(sess, os.path.join(SAVE_PATH, 'model'), global_step=batch_counter)
                    var_saver.save(sess, os.path.join(VAR_PATH, 'model'))

                    # Plot loss over time
                    fig = plt.figure()
                    plt.plot(test_steps, train_losses)
                    plt.savefig(os.path.join(logdir, 'train_loss.pdf'))
                    plt.close()

                    fig = plt.figure()
                    plt.plot(test_steps, test_losses)
                    plt.savefig(os.path.join(logdir, 'test_loss.pdf'))
                    plt.close()

                    helpers.graph_predicted_v_actual(NCOLS, config.params['OUT_NODES'], train_prediction, train_labels, 
                                                    train_colors, os.path.join(logdir, 'scatter_train.pdf'),
                                                    log_scale=LOG_SCALE)

                    helpers.graph_predicted_v_actual(NCOLS, config.params['OUT_NODES'], test_prediction, test_labels, 
                                                    test_colors, os.path.join(logdir, 'scatter_test.pdf'),
                                                    log_scale=LOG_SCALE)

                    helpers.graph_predicted_v_actual(NCOLS, config.params['OUT_NODES'], logfc_prediction, np.repeat(logfc_labels, config.params['OUT_NODES'], axis=1), 
                                                    logfc_colors, os.path.join(logdir, 'scatter_logfc.pdf'),
                                                    log_scale=False)

                    for i in range(config.params['OUT_NODES']):
                        logfc['predicted{}'.format(i)] = logfc_prediction[:, i]

                    logfc_agg = logfc.groupby(['miRNA_sequence', 'color']).agg(np.mean)

                    fig = plt.figure(figsize=(5*NCOLS,10))
                    for i in range(config.params['OUT_NODES']):
                        ax = plt.subplot(2,NCOLS,i+1)
                        ax.scatter(logfc_agg['predicted{}'.format(i)], logfc_agg['logFC'], alpha=0.5, s=20)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')

                    plt.tight_layout()
                    plt.savefig(os.path.join(logdir, 'scatter_logfc_agg.pdf'))
                    plt.close()


                    conv_weights = tf.get_collection('weight', scope='conv4x4')
                    conv_weights = sess.run(conv_weights)[0]
                    xlabels = ['U','A','G','C']
                    ylabels = ['A','U','C','G']
                    helpers.graph_convolutions(conv_weights, xlabels, ylabels, os.path.join(logdir, 'convolution1.pdf'))

                    conv_weights = tf.get_collection('weight', scope='convlayer2')
                    conv_weights = sess.run(conv_weights)[0]
                    helpers.graph_convolutions(conv_weights, None, None, os.path.join(logdir, 'convolution2.pdf'))

                    # conv_weights = tf.get_collection('weight', scope='convlayer3')
                    # conv_weights = sess.run(conv_weights)[0]
                    # helpers.graph_convolutions(conv_weights, None, None, os.path.join(logdir, 'convolution3.pdf'))
        reader.close()


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-i", "--infile", dest="INFILE", help="training data")
    parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs")
    parser.add_option("-n", dest="NUM_RUNS", type="int")

    (options, args) = parser.parse_args()

    metafile = open(os.path.join(options.LOGDIR, 'params.txt'), 'w')
    train_model(options.LOGDIR, os.path.join(options.LOGDIR, 'shuffled.txt'), metafile)
    metafile.close()

    # for i in range(options.NUM_RUNS):
    #     current_dir = os.path.join(options.LOGDIR, str(i))
    #     shuffled_file = os.path.join(current_dir, 'shuffled.txt')

    #     if not os.path.isdir(current_dir):
    #         os.makedirs(current_dir)
    #         helpers.shuffle_file(options.INFILE, shuffled_file)
    #     # else:
    #     #     print('{} already exists'.format(current_dir))
    #     #     continue

    #     metafile = open(os.path.join(current_dir, 'params.txt'), 'w')
    #     train_model(current_dir, shuffled_file, metafile)

    #     metafile.close()
