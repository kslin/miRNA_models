import numpy as np

import helpers

# Set global parameters
MIRLEN = 10
SEQLEN = 12
BATCH_SIZE_BIOCHEM = 10
BATCH_SIZE_REPRESSION = 25
KEEP_PROB_TRAIN = 0.5
STARTING_LEARNING_RATE = 0.001
LAMBDA = 0.001
NUM_EPOCHS = 50
REPORT_INT = 50
REPRESSION_WEIGHT = 50.0
# ZERO_OFFSET = 0
NORM_RATIO = 10.0
SWITCH_EPOCH = 25
UTR_COEF_INIT = -8.6379089
# DECAY_INIT = -0.6194858
DECAY_INIT = 0.0
LET7_INIT = -7.48320436

MIR_NTS = np.array(['A','T','C','G'])
SEQ_NTS = np.array(['T','A','G','C'])

MIR_NT_DICT = {nt:ix for (ix, nt) in enumerate(MIR_NTS)}
SEQ_NT_DICT = {nt:ix for (ix, nt) in enumerate(SEQ_NTS)}
TARGETS = np.eye(4)

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

SITE_DICT = {x: helpers.rev_comp(y[1:8]) + 'A' for (x,y) in MIRSEQ_DICT.items()}
print(SITE_DICT['let7'])

# make dictionary of reverse miRNA sequences trimmed to MIRLEN
MIRSEQ_DICT_MIRLEN = {x: y[:MIRLEN][::-1] for (x,y) in MIRSEQ_DICT.items()}
ONE_HOT_DICT = {x: helpers.one_hot_encode(y, MIR_NT_DICT, TARGETS) for (x,y) in MIRSEQ_DICT_MIRLEN.items()}
