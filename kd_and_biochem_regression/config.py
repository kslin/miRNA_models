import numpy as np

# Set global parameters
MIRLEN = 10
SEQLEN = 12
SEQ_BUFFER = 0

ONLY_CANON = False
OVERLAP_DIST = 7
BATCH_SIZE_BIOCHEM = 12
BATCH_SIZE_REPRESSION = 32
BATCH_SIZE_REPRESSION_TEST = 99
KEEP_PROB_TRAIN = 0.5
STARTING_LEARNING_RATE = 0.002
LAMBDA = 0.001
NUM_EPOCHS = 200
REPRESSION_WEIGHT = 50.0 * BATCH_SIZE_REPRESSION / BATCH_SIZE_BIOCHEM
SWITCH_EPOCH = 25
# UTR_COEF_INIT = -10.0
# DECAY_INIT = -1
# FREEAGO_INIT = -3.0
UTR_COEF_INIT = -8.5
DECAY_INIT = 0
FREEAGO_INIT = -4.0
PASS_OFFSET_INIT = -1.0
GUIDE_OFFSET_INIT = 0.0
HIDDEN1 = 4
HIDDEN2 = 16
HIDDEN3 = 16

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

MIRS5 = ['mir1','mir124','mir155','mir7','lsy6']
MIRS6 = MIRS5 + ['let7']
MIRS11 = sorted(['mir153','mir139','mir144','mir223','mir137',
             'mir205','mir143','mir182','mir199a','mir204','mir216b'])
MIRS12 = MIRS11 + ['let7']
MIRS16 = MIRS5 + MIRS11
MIRS17 = MIRS16 + ['let7']

# make dictionary of reverse miRNA sequences trimmed to MIRLEN
MIRSEQ_DICT_MIRLEN = {x: y[:MIRLEN][::-1] for (x,y) in MIRSEQ_DICT.items()}
