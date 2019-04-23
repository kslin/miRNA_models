from optparse import OptionParser
import itertools as it

import numpy as np
import pandas as pd
import tensorflow as tf

import utils


def generate_12mers(site8):
    random_8mers = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"],repeat=8))]
    mers = []
    for i in range(5):
        subseq = site8[i:i+4]
        mers += [x[:i+2] + subseq + x[i+2:] for x in random_8mers]
    mers = list(set(mers))
    return sorted(mers)


def calculate_12mer_kds(mirname, mirseq, mirlen, load_model, outfile):
    """
    For a given miRNA sequence, use a saved ConvNet to generate predictions
    """

    if len(mirseq) < 12:
        raise(ValueError("miRNA must be at least 12 nt long"))
    
    if len(mirseq.replace('A','').replace('C','').replace('G','').replace('T','')) > 0:
        raise(ValueError("miRNA must only contain A, C, T, G"))

    site8 = utils.rev_comp(mirseq[1:8]) + 'A'
    mirseq_one_hot = utils.one_hot_encode(mirseq[:mirlen])

    # generate all 12mer sequences, there should be 262,144
    kmers = generate_12mers(site8)

    if len(kmers) != 262144:
        raise(ValueError("kmers should be 262144 in length"))

    # load trained model 
    tf.reset_default_graph()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(load_model + '.meta')
        saver.restore(sess, load_model)

        _dropout_rate = tf.get_default_graph().get_tensor_by_name('dropout_rate:0')
        _phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        _combined_x = tf.get_default_graph().get_tensor_by_name('combined_x:0')
        _prediction = tf.get_default_graph().get_tensor_by_name('final_layer/pred_ka:0')

        num_batches = 64
        batch_size = 4096

        with open(outfile, 'w') as outfile_writer:
            outfile_writer.write('12mer\tlog_kd\tmir\tmirseq\tstype\n')
            for batch in range(num_batches):
                print("Processing {}/{}...".format((batch+1)*batch_size, 262144))
                seqs = kmers[batch*batch_size: (batch+1) * batch_size]
                input_data = []
                for ix, seq in enumerate(seqs):
                    seq_one_hot = utils.one_hot_encode(seq)
                    input_data.append(np.outer(mirseq_one_hot, seq_one_hot))

                input_data = np.stack(input_data)

                feed_dict = {
                                _dropout_rate: 0.0,
                                _phase_train: False,
                                _combined_x: input_data
                            }

                pred_kds = -1 * sess.run(_prediction, feed_dict=feed_dict).flatten()
                stypes = [utils.get_centered_stype(site8, seq) for seq in seqs]

                for seq, kd, stype in zip(seqs, pred_kds, stypes):
                    outfile_writer.write('{}\t{}\t{}\t{}\t{}\n'.format(seq, kd, mirname, mirseq, stype))


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--name", dest="MIRNAME", help="miRNA name")
    parser.add_option("--mirseq", dest="MIRSEQ", help="miRNA sequence", default=None)
    parser.add_option("--mirlen", dest="MIRLEN", type=int)
    parser.add_option("--load_model", dest="LOAD_MODEL", help="trained model to use")
    parser.add_option("--outfile", dest="OUTFILE", help="output file")

    (options, args) = parser.parse_args()

    if options.MIRSEQ is not None:
        calculate_12mer_kds(options.MIRNAME, options.MIRSEQ, options.MIRLEN, options.LOAD_MODEL, options.OUTFILE)

    else:
        mirdata = pd.read_csv(options.MIRNAME, sep='\t')
        for row in mirdata.iterrows():
            mirname, mirseq = row[1]['mir'], row[1]['guide_seq']
            print(mirname, mirseq)
            calculate_12mer_kds(mirname, mirseq, options.MIRLEN,
                options.LOAD_MODEL, options.OUTFILE.replace('MIR', mirname))

