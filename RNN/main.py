from optparse import OptionParser
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, static_bidirectional_rnn

import helpers
# from model import inference
# import train_model


if __name__ == '__main__':

    num_hidden = 32
    # num_layers = 3
    length = 50
    # num_classes = 1
    num_nts = 5
    batch_size = 20

    data = tf.placeholder(tf.float32, [None, length, num_nts])
    target = tf.placeholder(tf.float32, [None, length])
    keep_prob = tf.placeholder(tf.float32)

    lstm_fw_cell = BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_bw_cell = BasicLSTMCell(num_hidden, forget_bias=1.0)

    x = tf.unstack(data, length, 1)

    outputs, _, _ = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    weight = helpers.weight_variable([2*num_hidden, 1])
    bias = helpers.bias_variable([1])

    predictions = tf.stack([tf.nn.tanh(tf.matmul(out, weight) + bias) for out in outputs], axis=1)

    predictions = tf.reshape(predictions, [-1, length])

    loss = tf.nn.l2_loss(tf.subtract(predictions, target))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    ### DMS DATA

    dms = pd.read_csv('/lab/bartel4_ata/kathyl/RNA_Seq/data/accessibility/dms_vals.txt',sep='\t',header=None)

    nts = ',N,'.join(list(dms[0]))
    vals = ',0.0,'.join(list(dms[1]))
    nts = helpers.one_hot_encode_nt(''.join(nts.split(',')), np.array(['A','T','C','G','N']))
    vals = np.array([float(x) for x in vals.split(',')]) * 20 - 1.0

    train_x = nts[:-2000, :]
    train_y = vals[:-2000]

    test_x = nts[-2000:, :]
    test_y = vals[-2000:]


    test_x = np.zeros((40,length,num_nts))
    test_y = np.zeros((40,length))
    ix = 0
    for i in range(40):
        test_x[i,:,:] = test_x[ix*length: (ix+1)*length, :]
        test_y[i,:] = test_y[ix*length: (ix+1)*length]
        ix += 1

    ###

    ### RNAFOLD DATA ###

    # temp_folder = '/lab/bartel4_ata/kathyl/NeuralNet/temp'

    # if not os.path.isdir(temp_folder):
    #     os.makedirs(temp_folder)

    # utrs = pd.read_csv('/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/3pseq/utr3.txt',sep='\t',index_col=0)
    # train_x = []
    # train_y = []
    # test_x = []
    # test_y = []
    # FOLD_LEN = 8
    # t0 = time.time()
    # for ix, row in enumerate(utrs.iterrows()):
    #     seq = row[1]['Sequence']
    #     if len(seq) > 1000:
    #         seq = seq[:1000]

    #     if ix > 100:
    #         train_y.append(helpers.get_rnaplfold_data(row[0], seq, temp_folder, FOLD_LEN)[[FOLD_LEN]].fillna(0).values)
    #         train_y.append([[0.5]])
    #         train_x.append(helpers.one_hot_encode_nt(seq + 'N', np.array(['A','T','C','G','N'])))
    #     else:
    #         test_y.append(helpers.get_rnaplfold_data(row[0], seq, temp_folder, FOLD_LEN)[[FOLD_LEN]].fillna(0).values)
    #         test_y.append([[0.5]])
    #         test_x.append(helpers.one_hot_encode_nt(seq + 'N', np.array(['A','T','C','G','N'])))


    # print(ix)
    # print(time.time() - t0)

    # train_x = np.concatenate(train_x)
    # train_y = np.concatenate(train_y)
    # test_x = np.concatenate(test_x)
    # test_y = np.concatenate(test_y)

    # print(train_x.shape, train_y.shape)
    # print(test_x.shape, test_y.shape)

    ###


    

    max_steps = 2000

    actual = test_y.flatten()

    steps = []
    losses = []

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        epoch_counter = 1
        ix = 0
        for num_steps in range(max_steps):

            # when end of seq is reaches, start another epoch
            if ((ix+batch_size)*length < len(train_x)):
                print('Finished epoch {}'.format(epoch_counter))
                epoch_counter += 1
                ix = 0

            batch_train_x = np.zeros((batch_size,length,num_nts))
            batch_train_y = np.zeros((batch_size,length))
            for i in range(batch_size):
                batch_train_x[i,:,:] = train_x[ix*length: (ix+1)*length, :]
                batch_train_y[i,:] = train_y[ix*length: (ix+1)*length]
                ix += 1

            _ = sess.run(train_step, feed_dict={
                                                data:batch_train_x,
                                                target: batch_train_y,
                                                keep_prob: 0.5
                })


            if (num_steps % 100) == 0:
                print(num_steps)
                test_loss, preds = sess.run([loss, predictions], feed_dict={data:test_x, target: test_y, keep_prob: 1.0})
                preds = preds.flatten()
                steps.append(num_steps)
                losses.append(test_loss)

                fig = plt.figure(figsize=(14,7))
                plt.plot(steps, losses)
                plt.savefig('losses_bidirec.png')
                plt.close()


                fig = plt.figure(figsize=(7,7))
                plt.scatter(preds, actual)
                plt.savefig('prediction_bidirec.png')
                plt.close()




    

    # print(output)
    # prediction = tf.reshape(output, [-1, 1])

    # cells = []
    # for _ in range(num_layers):
    #     cell = tf.contrib.rnn.LSTMCell(num_hidden)  # Or LSTMCell(num_hidden)
    #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    #     cells.append(cell)
    # cell = tf.contrib.rnn.MultiRNNCell(cells)

    # data = tf.placeholder(tf.float32, [None, None, 4])
    # output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

    # out_size = target.get_shape()[2].value
    # logit = tf.contrib.layers.fully_connected(output, out_size, activation_fn=None)
    # prediction = tf.nn.softmax(logit)




    # parser = OptionParser()
    # parser.add_option("-i", "--infile", dest="INFILE", help="training data")
    # parser.add_option("-f", "--logfc_file", dest="LOGFC_FILE", help="logfc data")
    # parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory for writing logs", default=None)

    # (options, args) = parser.parse_args()

    # params = {
    #             'MIRLEN': 10,
    #             'SEQLEN': 12,
    #             'IN_NODES': 1,
    #             'OUT_NODES': 1,
    #             'HIDDEN1': 4,
    #             # 'HIDDEN2': 8,
    #             'HIDDEN3': 16,
    #             'ERROR_MODEL': 'l2',
    #             'MAX_EPOCH': 10,
    #             'BATCH_SIZE': 200,
    #             'LOGFC_BATCH_SIZE': 200,
    #             'REPORT_INT': 100,
    #             'KEEP_PROB_TRAIN': 0.5,
    #             'TEST_SIZE': 5000,
    #             'RESTORE_FROM': None,
    #             'NUM_RUNS': 2,
    #             'STARTING_LEARNING_RATE': 0.002,
    #             'LAMBDA': 0.00005,
    #             'LOG_SCALE': False,
    #             'NCOLS': 1
    #     }

    # print(params)

    # if not os.path.isdir(options.LOGDIR):
    #     os.makedirs(options.LOGDIR)

    # metafile = open(os.path.join(options.LOGDIR, 'params.txt'), 'w')
    # for key in sorted(params.keys()):
    #     metafile.write('{}: {}\n'.format(key, params[key]))

    # # read in kd data:
    # data = pd.read_csv(options.INFILE, header=None)
    # data.columns = ['mirseq','seq','kd']
    # data['sitem8'] = [helpers.complementaryT(mirseq[-8:-1]) for mirseq in data['mirseq']]
    # data['color'] = [helpers.get_color_old(sitem8, seq) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]
    # data['color2'] = [helpers.get_color_old(sitem8, seq[2:10]) for (sitem8, seq) in zip(data['sitem8'], data['seq'])]

    # # get rid of sequences with sites out of register
    # print(len(data))
    # data = data[data['color'] == data['color2']].drop('color2',1)
    # print(len(data))

    # # log-transformation and zero-centering
    # data['log_kd'] = (-1*np.log2(data['kd']) - 4)/6.4

    # shuffle_ix = np.random.permutation(len(data))
    # data = data.iloc[shuffle_ix]

    # # read in logfc data
    # logfc = pd.read_csv(options.LOGFC_FILE, sep='\t')
    # logfc['sitem8'] = [helpers.complementaryT(mirseq[-8:-1]) for mirseq in logfc['mirseq']]
    # logfc['color'] = [helpers.get_color_old(sitem8, seq) for (sitem8, seq) in zip(logfc['sitem8'], logfc['seq'])]
    # logfc['logFC'] /= -1

    # print(len(data), len(logfc))

    # # train_hyperparam(data, logfc, params, inference)

    # # train_kd_only(data, logfc, params, inference, os.path.join(options.LOGDIR, 'kd_only'))

    # all_mirs = logfc['mir'].unique()
    # all_mirs = [a for a in all_mirs if a not in ['let7']]
    # possible_mirs = [a for a in all_mirs if a not in ['mir1','mir124','mir155','lsy6','fake_mir']]

    # tested = []

    # for i in range(10):
    #     while True:
    #         test_mirs = sorted(list(np.random.choice(possible_mirs, size=3, replace=False)))
    #         dir_name = '_'.join(test_mirs)
    #         if dir_name not in tested:
    #             break

    #     tested.append(dir_name)

    #     train_mirs = [a for a in all_mirs if a not in test_mirs]
    #     print(test_mirs)
    #     print(train_mirs)

    #     print(os.path.join(options.LOGDIR, '_'.join(test_mirs)))

    #     train_logfc_and_kd(data, logfc, params, train_mirs, test_mirs, inference,
    #                              os.path.join(options.LOGDIR, dir_name))