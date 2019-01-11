import itertools as it

import numpy as np
import tensorflow as tf

import config, helpers


def generate_12mers(sitem8):
    random_8mers = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"],repeat=8))]
    mers = []
    for i in range(5):
        subseq = sitem8[i:i+4]
        mers += [x[:i+2] + subseq + x[i+2:] for x in random_8mers]
    mers = list(set(mers))
    return sorted(mers)


def calculate_12mer_kds(mirseq, mirname, logdir, outfile):
    """
    For a given miRNA sequence, use a saved ConvNet to generate predictions
    """

    if len(mirseq) < 12:
        raise(ValueError("miRNA must be at least 12 nt long"))
    
    if len(mirseq.replace('A','').replace('C','').replace('G','').replace('T','')) > 0:
        raise(ValueError("miRNA must only contain A, C, T, G"))

    sitem8 = helpers.rev_comp(mirseq[1:8]) + 'A'
    mirseq_one_hot = helpers.one_hot_encode(mirseq[:config.MIRLEN][::-1], config.MIR_NT_DICT, config.TARGETS)

    # generate all 12mer sequences, there should be 262,144
    kmers = generate_12mers(sitem8)

    if len(kmers) != 262144:
        raise(ValueError("kmers should be 262144 in length"))

    # load trained model 
    tf.reset_default_graph()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        latest = tf.train.latest_checkpoint(logdir)
        print("Loading model from {}".format(latest))
        saver = tf.train.import_meta_graph(latest + '.meta')
        saver.restore(sess, latest)

        _keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        _phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        _combined_x = tf.get_default_graph().get_tensor_by_name('biochem_x:0')
        _prediction = tf.get_default_graph().get_tensor_by_name('final_layer/pred_ka:0')

        num_batches = 64
        batch_size = 4096

        with open(outfile, 'w') as outfile_writer:
            outfile_writer.write('12mer\tlog_kd\tmir\tmirseq\tstype\n')
            for batch in range(num_batches):
                # print("Processing {}/{}...".format((batch+1)*batch_size, 262144))
                seqs = kmers[batch*batch_size: (batch+1) * batch_size]
                input_data = np.zeros([batch_size, 4*config.MIRLEN, 4*config.SEQLEN])
                for ix, seq in enumerate(seqs):
                    seq_one_hot = helpers.one_hot_encode(seq, config.SEQ_NT_DICT, config.TARGETS)
                    input_data[ix,:,:] = np.outer(mirseq_one_hot, seq_one_hot)

                input_data = np.expand_dims((input_data * 4) - 0.25, 3)


                feed_dict = {
                                _keep_prob: 1.0,
                                _phase_train: False,
                                _combined_x: input_data
                            }

                pred_kds = -1 * sess.run(_prediction, feed_dict=feed_dict).flatten()
                stypes = [helpers.get_stype_six_canon(sitem8, seq) for seq in seqs]

                for seq, kd, stype in zip(seqs, pred_kds, stypes):
                    outfile_writer.write('{}\t{}\t{}\t{}\t{}\n'.format(seq, kd, mirname, mirseq, stype))


if __name__ == '__main__':

    # parser = OptionParser()
    # parser.add_option("-m", "--mirseq", dest="MIRSEQ", help="miRNA sequence")
    # parser.add_option("-n", "--name", dest="MIRNAME", help="miRNA name")
    # parser.add_option("-l", "--logdir", dest="LOGDIR", help="directory with saved model")
    # parser.add_option("-o", "--outfile", dest="OUTFILE", help="output file")

    # (options, args) = parser.parse_args()

    # calculate_12mer_kds(options.MIRSEQ, options.MIRNAME, options.LOGDIR, options.OUTFILE)

    logdir = '/lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_neg_examples/MIRNA/saved/'
    outfile = '/lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/kd_preds/simple_xval_4_16_16_neg_examples/MIRNA.txt'

    ALL_MIRS = ['lsy6', 'let7', 'mir1', 'mir7', 'mir124', 'mir137', 'mir139', 'mir143', 'mir144',
                'mir153', 'mir155', 'mir182', 'mir199a', 'mir204', 'mir205', 'mir216b', 'mir223']

    for mir in ALL_MIRS:
        print(mir)
        mirseq = config.MIRSEQ_DICT[mir]
        calculate_12mer_kds(mirseq, mir, logdir.replace('MIRNA', mir), outfile.replace('MIRNA', mir))

        mirseq_star = config.MIRSEQ_DICT[mir + '*']
        calculate_12mer_kds(mirseq_star, mir+'*', logdir.replace('MIRNA', mir), outfile.replace('MIRNA', mir+'_pass'))

