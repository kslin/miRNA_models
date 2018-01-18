import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson
import tensorflow as tf

import helpers

def make_example(length, seq, labels, mask):
    # make a SequenceExample instance
    ex = tf.train.SequenceExample()
    ex.context.feature["length"].int64_list.value.append(length)
    fl_nts = ex.feature_lists.feature_list["nts"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    fl_mask = ex.feature_lists.feature_list["mask"]

    for seq_nt, label, m in zip(seq, labels, mask):
        for nt in ['A','T','C','G']:
            fl_nts.feature.add().float_list.value.append(float(nt == seq_nt) - 0.25)
        fl_labels.feature.add().float_list.value.append(label)
        fl_mask.feature.add().float_list.value.append(m)

    return ex


def make_motif_label(length, seq, motif):
    if motif not in seq:
        label = [0.0]*length

    else:
        locs = np.array([m.start() for m in re.finditer('(?={})'.format(motif), seq)])
        label = [(2*np.min(np.abs(locs - ix))/length) - 1.0 for ix in range(length)]

    return label

         
def make_rnaplfold_label(length, seq, temp_folder, fold_len):
    label = (2*helpers.get_rnaplfold_data('A', seq, temp_folder, fold_len)[fold_len].fillna(0).values) - 1

    return label


def write_generated_records(mean_len, num_examples, label_func, outfile_name):
    tfrecord_writer = tf.python_io.TFRecordWriter(outfile_name + '.tfrecords')
    plain_writer = open(outfile_name + '.txt', 'w')

    lengths = poisson.rvs(mean_len, size=num_examples)

    for length in lengths:
        seq = helpers.generate_random_seq(length)
        mask = [1]*length
        label = label_func(length, seq)
        ex = make_example(length, seq, label, mask)
        tfrecord_writer.write(ex.SerializeToString())
        plain_writer.write(seq + '\n')
        plain_writer.write(','.join([str(x) for x in label]) + '\n')

    tfrecord_writer.close()
    plain_writer.close()


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "nts": tf.FixedLenSequenceFeature([], dtype=tf.float32),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.float32),
        "mask": tf.FixedLenSequenceFeature([], dtype=tf.float32)
    }

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    # extract length
    length = tf.cast(features[0]['length'], tf.int32)

    # extract sequence
    seq = tf.cast(features[1]['nts'], tf.float32)

    # extract labels
    labels = tf.cast(features[1]['labels'], tf.float32)

    # extract mask
    mask = tf.cast(features[1]['mask'], tf.float32)

    return length, seq, labels, mask


def get_inputs(infile, istraining, batch_size, num_epochs):

    if istraining:
        filename_queue = tf.train.string_input_producer([infile], num_epochs=num_epochs)
    else:
        filename_queue = tf.train.string_input_producer([infile], num_epochs=None)


    batched_data = read_and_decode(filename_queue)
    batched_data = tf.train.batch(
                                batched_data,
                                dynamic_pad=True,
                                batch_size=batch_size,
                                num_threads=1,
                                capacity=1000 + 3 * batch_size,
                            )

    return batched_data


def train_rnn(train_file, val_file):
    num_hidden = 64
    num_nts = 4
    batch_size = 200
    num_epochs = 100
    val_size = 100
    starting_learning_rate = 0.01
    report_step = 50

    tf.reset_default_graph()

    with tf.Session() as sess:

        cell_fw = tf.nn.rnn_cell.LSTMCell(
                                            num_units=num_hidden,
                                            state_is_tuple=True,
                                            initializer=tf.truncated_normal_initializer
                )

        cell_bw = tf.nn.rnn_cell.LSTMCell(
                                            num_units=num_hidden,
                                            state_is_tuple=True,
                                            initializer=tf.truncated_normal_initializer
                )

        weight = helpers.weight_variable([2*num_hidden, 1], name='weight')
        bias = helpers.bias_variable([1], name='bias')

        # train graph
        train_lengths, train_seqs, train_labels, train_mask = get_inputs(
                                                                            train_file,
                                                                            istraining=True,
                                                                            batch_size=batch_size,
                                                                            num_epochs=num_epochs
                                                                        )

        train_mask_flat = tf.reshape(train_mask, [-1,1])
        train_labels_flat = tf.reshape(train_labels, [-1,1])
        train_batch_max_length = tf.reduce_max(train_lengths)
        train_seqs_image = tf.reshape(train_seqs, [-1, train_batch_max_length, num_nts])        
 
        train_outputs, train_states  = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                dtype=tf.float32,
                sequence_length=train_lengths,
                inputs=train_seqs_image
            )

        train_outputs_flat = tf.reshape(tf.concat(train_outputs, 2), [-1, 2*num_hidden])
        train_predictions = tf.nn.tanh(tf.matmul(train_outputs_flat, weight) + bias)
        train_loss = tf.nn.l2_loss(tf.multiply(tf.subtract(train_predictions, train_labels_flat), train_mask_flat))

        optimizer = tf.train.AdamOptimizer(learning_rate=starting_learning_rate)
        model_vars = tf.trainable_variables()
        grads = tf.gradients(train_loss, model_vars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, 1)
        train_step = optimizer.apply_gradients(zip(clipped_grads, model_vars))
        # train_step = tf.train.AdamOptimizer(starting_learning_rate).minimize(train_loss)

        # val graph
        val_lengths, val_seqs, val_labels, val_mask = get_inputs(
                                                                            val_file,
                                                                            istraining=False,
                                                                            batch_size=val_size,
                                                                            num_epochs=None
                                                                        )

        val_mask_flat = tf.reshape(val_mask, [-1,1])
        val_labels_flat = tf.reshape(val_labels, [-1,1])
        val_batch_max_length = tf.reduce_max(val_lengths)
        val_seqs_image = tf.reshape(val_seqs, [-1, val_batch_max_length, num_nts])        
 
        val_outputs, val_states  = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                dtype=tf.float32,
                sequence_length=val_lengths,
                inputs=val_seqs_image
            )

        val_outputs_flat = tf.reshape(tf.concat(val_outputs, 2), [-1, 2*num_hidden])
        val_predictions = tf.nn.tanh(tf.matmul(val_outputs_flat, weight) + bias)
        val_loss = tf.nn.l2_loss(tf.multiply(tf.subtract(val_predictions, val_labels_flat), val_mask_flat))

        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        all_steps = []
        all_losses = []
        try:
            step = 0
            while not coord.should_stop():
                

                if (step % report_step) == 0:
                    xs, ys, val_loss_num, val_pred = sess.run([val_predictions, val_labels_flat, val_loss, val_predictions])
                    # print(val_loss_num)
                    # print(np.sum(val_pred))
                    plt.scatter(xs,ys, s=10)
                    plt.savefig('training_decay.png')
                    plt.close()

                    all_steps.append(step)
                    all_losses.append(val_loss_num)

                    plt.plot(all_steps, all_losses)
                    plt.savefig('test_loss.png')
                    plt.close()
                    # print(g[-1])
                    # print(c[-1])
                #     val_labels_num, val_predictions_num, val_loss_num, p = sess.run([val_labels, val_predictions, val_loss, val_seqs_image])
                #     print(val_predictions_num[0])
                #     print(p[0])
                    # print(val_predictions_num.shape)
                    # print(step)
                    # firstone = np.argwhere(val_labels_num[0] == 1.0)[0]

                    # print(val_labels_num[0][:10])
                    # print(val_predictions_num[0][:10])

                _, v, l, c = sess.run([train_step, train_labels, train_loss, train_states])
                    # print(np.sum(c[0]))
                # for i,row in enumerate(t_labels):
                #     if np.isnan(np.sum(row)):
                #         print(i)
                #         nseq = int(len(t_seqs[i])/4)
                #         print(row)
                #         # print(t_seqs[i].reshape((nseq, 4)))
                #         break

                
                # print(cell)

                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training')

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)


if __name__ == '__main__':

    # label_func = lambda x,y: make_motif_label(x, y, 'AG')
    # write_generated_records(100, 4000, label_func, 'training')
    # write_generated_records(100, 100, label_func, 'testing')

    # temp_folder = '/lab/bartel4_ata/kathyl/NeuralNet/temp'

    # if not os.path.isdir(temp_folder):
    #     os.makedirs(temp_folder)

    # label_func = lambda x,y: make_rnaplfold_label(x, y, temp_folder, 8)
    # write_generated_records(100, 10000, label_func, 'training_rnaplfold')
    # write_generated_records(100, 100, label_func, 'testing_rnaplfold')


    # train_rnn(train_file="training.tfrecords", val_file="testing.tfrecords")
    train_rnn(train_file="training_rnaplfold.tfrecords", val_file="testing_rnaplfold.tfrecords")

    



        



        

