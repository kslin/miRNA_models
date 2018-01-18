import itertools as it
import os
import subprocess

import helpers

def generate_duplex():

    new_mirseq = [helpers.generate_random_seq(20) for _ in range(5)]
    up_flank = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"],repeat=5))]
    down_flank = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"],repeat=3))]

    mirseqs, seqs = [], []
    file1 = '/lab/bartel4_ata/kathyl/NeuralNet/data/simple/for_rnaduplex.txt'
    file2 = '/lab/bartel4_ata/kathyl/NeuralNet/data/simple/rnaduplex_output.txt'
    file3 = '/lab/bartel4_ata/kathyl/NeuralNet/data/simple/rnaduplex_features.txt'
    with open(file1,'w') as outfile:
        # outfile.write('mirseq\tseq\n')
        for mirseq in new_mirseq:
            print(mirseq)
            site = helpers.complementary(mirseq[-5:-1])
            for up in up_flank:
                for down in down_flank:
                    outfile.write('{}\n{}\n'.format(mirseq, up+site+down))
                    mirseqs.append(mirseq)
                    seqs.append(up+site+down)


    mycall = 'RNAduplex < {} > {}'.format(file1, file2)
    subprocess.call([mycall], shell=True, stdout=subprocess.PIPE)

    results = open(file2, 'r')
    with open(file3, 'w') as outfile:
        for mirseq, seq in zip(mirseqs, seqs):
            res = 2**float(results.readline().split()[-1][1:-1])
            outfile.write('{},{},{}\n'.format(mirseq, seq, str(res)))

    results.close()
    os.remove(file2)


def generate_duplex_logfc():

    new_mirseq = [helpers.generate_random_seq(20) for _ in range(12)]
    up_flank = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"],repeat=5))]
    down_flank = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"],repeat=3))]

    mirseqs, seqs = [], []
    file1 = '/lab/bartel4_ata/kathyl/NeuralNet/data/simple/logfc_for_rnaduplex.txt'
    file2 = '/lab/bartel4_ata/kathyl/NeuralNet/data/simple/logfc_rnaduplex_output.txt'
    file3 = '/lab/bartel4_ata/kathyl/NeuralNet/data/simple/logfc_rnaduplex_features.txt'
    with open(file1,'w') as outfile:
        for mirseq in new_mirseq:
            print(mirseq)
            site = helpers.complementary(mirseq[-5:-1])
            for up in up_flank:
                for down in down_flank:
                    outfile.write('{}\n{}\n'.format(mirseq, up+site+down))
                    mirseqs.append(mirseq)
                    seqs.append(up+site+down)


    mycall = 'RNAduplex < {} > {}'.format(file1, file2)
    subprocess.call([mycall], shell=True, stdout=subprocess.PIPE)

    results = open(file2, 'r')
    with open(file3, 'w') as outfile:
        outfile.write('mirseq\tseq\tlogFC\tmir\n')
        for mirseq, seq in zip(mirseqs, seqs):
            res = float(results.readline().split()[-1][1:-1])
            outfile.write('{}\t{}\t{}\t{}\n'.format(mirseq, seq, str(res),'mir1'))

    results.close()
    os.remove(file2)


generate_duplex_logfc()
