import numpy as np

import helpers

def num_match(seq1, seq2):
    matches = 0
    for s1, s2 in zip(seq1[::-1], seq2[::-1]):
        matches += (s1 == s2)

    return matches

def generate_random_pairs():

    mirseq = helpers.generate_random_seq(20)
    sitem8 = helpers.complementaryT(mirseq[-8:-1])
    seq = helpers.generate_random_seq(12)

    r = np.random.randint(7)
    if r > 0:
        seq = helpers.generate_random_seq(12-r) + sitem8[-r:]

    # matches = []
    # for i in range(len(seq)):
    #     subseq = seq[:i+1]
    #     matches.append(num_match(sitem8, subseq))

    # return mirseq, seq, max(matches)

    return mirseq, seq, r


for i in range(10):
    print(generate_random_pairs())
    
with open('/lab/bartel4_ata/kathyl/NeuralNet/data/simple/num_matches.txt','w') as outfile:
    for i in range(1000000):
        mirseq, seq, nmatches = generate_random_pairs()
        outfile.write('{},{},0,0,{}\n'.format(mirseq, seq, np.exp(nmatches + np.random.normal(scale=0.2))))
        # outfile.write('{},{},0,0,{}\n'.format(mirseq, seq, nmatches))