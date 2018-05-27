import numpy as np

import helpers

# Set global parameters
MIRLEN = 12
SEQLEN = 12
BATCH_SIZE_BIOCHEM = 100
BATCH_SIZE_REPRESSION = 50
KEEP_PROB_TRAIN = 0.5
STARTING_LEARNING_RATE = 0.001
LAMBDA = 0.001
NUM_EPOCHS = 50
REPORT_INT = 50
REPRESSION_WEIGHT = 10.0
ZERO_OFFSET = 0
NORM_RATIO = 10.0
SWITCH_EPOCH = -1

# HIDDEN1 = 4
# HIDDEN2 = 8
# HIDDEN3 = 16

MIR_NTS = np.array(['A','T','C','G'])
SEQ_NTS = np.array(['T','A','G','C'])

MIRSEQ_DICT = {
                  'mir137': 'TTATTGCTTAAGAATACGCGTAG',
                  'mir137*': 'ACGCGTATTCTTAAGCAATAAAT',
                  'mir205': 'TCCTTCATTCCACCGGAGTCTG',
                  'mir205*': 'GACTCCGGTGGAATGAAGCAAT',
                  'mir155': 'TTAATGCTAATCGTGATAGGGGT',
                  'mir155*': 'CCCTATCACGATTAGCATTAAAT',
                  'mir223': 'TGTCAGTTTGTCAAATACCCCA',
                  'mir223*': 'GGGTATTTGACAAACTGATAAT',
                  'mir144': 'TACAGTATAGATGATGTACT',
                  'mir144*': 'TACATCATCTATACTCTAAT',
                  'mir143': 'TGAGATGAAGCACTGTAGCTC',
                  'mir143*': 'GCTACAGTGCTTCATCTTAAT',
                  'mir153': 'TTGCATAGTCACAAAAGTGATC',
                  'mir153*': 'TCACTTTTGTGACTATGTAAAT',
                  'mir216b': 'AAATCTCTGCAGGCAAATGTGA',
                  'mir216b*': 'ACATTTGCCTGCAGAGATTTAT',
                  'mir199a': 'CCCAGTGTTCAGACTACCTGTTC',
                  'mir199a*': 'ACAGGTAGTCTGAACACTGCGAT',
                  'mir204': 'TTCCCTTTGTCATCCTATGCCT',
                  'mir204*': 'GCATAGGATGACAAAGGCAAAT',
                  'mir139': 'TCTACAGTGCACGTGTCTCCAGT',
                  'mir139*': 'TGGAGACACGTGCACTGTACAAT',
                  'mir182': 'TTTGGCAATGGTAGAACTCACACT',
                  'mir182*': 'TGTGAGTTCTACCATTGCTAAAAT',
                  'mir7': 'TGGAAGACTAGTGATTTTGTTGT',
                  'mir7*': 'AACAAAATCACTAGTCTTCTAAT',
                  'let7': 'TGAGGTAGTAGGTTGTATAGTT',
                  'let7*': 'CTATACAACCTACTACCTTAAT',
                  'mir1': 'TGGAATGTAAAGAAGTATGTAT',
                  'mir1*': 'ACATACTTCTTTACATTCTAAT',
                  'mir124': 'TAAGGCACGCGGTGAATGCCAA',
                  'mir124*': 'GGCATTCACCGCGTGCTTTAAT',
                  'lsy6': 'TTTTGTATGAGACGCATTTCGA',
                  'lsy6*': 'GAAATGCGTCTCATACAAAAAT',
                  'mir7-24nt': 'TGGAAGACTAGTGATTTTGTTGTT',
                  'mir7-25nt': 'TGGAAGACTAGTGATTTTGTTGTTT'
            }

SITE_DICT = {x: helpers.rev_comp(y[1:8]) for (x,y) in MIRSEQ_DICT.items()}

# make dictionary of reverse miRNA sequences trimmed to MIRLEN
MIRSEQ_DICT_MIRLEN = {x: y[:MIRLEN][::-1] for (x,y) in MIRSEQ_DICT.items()}
ONE_HOT_DICT = {x: helpers.one_hot_encode_nt_new(y, MIR_NTS) for (x,y) in MIRSEQ_DICT_MIRLEN.items()}

FREEAGO_SITE_DICT = {
                        'mir223': -3.16,
                        'mir153': -3.33,
                        'mir216b': -2.9,
                        'mir205': -3.14,
                        'mir139': -4.34,
                        'mir199a': -3.46,
                        'mir155': -4.19,
                        'mir1': -4.13,
                        'mir182': -4.09,
                        'mir204': -3.28,
                        'mir144': -5.5,
                        'mir137': -3.27,
                        'lsy6': -4.8,
                        'mir124': -4.7,
                        'mir143': -3.57,
                        'mir7': -4.76
                  }

FREEAGO_PASS_DICT = {
                        'mir223':  -7.03,
                        'mir153': -5.87,
                        'mir216b': -6.48,
                        'mir205': -7.46,
                        'mir139': -5.08,
                        'mir199a': -7.1,
                        'mir155': -7.74,
                        'mir1': -3.82,
                        'mir182': -4.66,
                        'mir204': -4.41,
                        'mir144': -5.8,
                        'mir137': -9.03,
                        'lsy6': -7.66,
                        'mir124': -10.7,
                        'mir143': -6.32,
                        'mir7': -9.47
                  }

# mir223: -3.16, -7.03
# mir153: -3.33, -5.87
# mir216b: -2.9, -6.48
# mir205: -3.14, -7.46
# mir139: -4.34, -5.08
# mir199a: -3.46, -7.1
# mir155: -4.19, -7.74
# mir1: -4.13, -3.82
# mir182: -4.09, -4.66
# mir204: -3.28, -4.41
# mir144: -5.5, -5.8
# mir137: -3.27, -9.03
# lsy6: -4.8, -7.66
# mir124: -4.7, -10.7
# mir143: -3.57, -6.32
# mir7: -4.76, -9.47