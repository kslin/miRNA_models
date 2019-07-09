import itertools as it
from optparse import OptionParser
import os

import numpy as np
import pandas as pd


def generate_12mers(site8, only_canon):
    mers = []
    if only_canon:
        all_6mers = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"], repeat=6))]
        for i in range(3):
            subseq = site8[i:i+6]
            mers += [x[:i+2] + subseq + x[i+2:] for x in all_6mers]

    else:
        all_8mers = ["".join(kmer) for kmer in list(it.product(["A","C","G","T"], repeat=8))]
        for i in range(5):
            subseq = site8[i:i+4]
            mers += [x[:i+2] + subseq + x[i+2:] for x in all_8mers]
    
    mers = list(set(mers))
    return sorted(mers)


def get_centered_stype(site8, seq):
    if site8 == seq[2:-2]:
        return '8mer'
    elif site8[:-1] == seq[2:-3]:
        return '7mer-m8'
    elif site8[1:] == seq[3:-2]:
        return '7mer-a1'
    elif site8[1:-1] == seq[3:-3]:
        return '6mer'
    elif site8[:-2] == seq[2:-4]:
        return '6mer-m8'
    elif site8[2:] == seq[4:-2]:
        return '6mer-a1'
    else:
        return 'no site'


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--mirseqs", dest="MIRSEQS", help="table with miRNA sequence information")
    parser.add_option("--outdir", dest="OUTDIR", help="folder for writing outputs")

    (options, args) = parser.parse_args()

    if (not os.path.isdir(options.OUTDIR)):
        os.makedirs(options.OUTDIR)

    MIRSEQS = pd.read_csv(options.MIRSEQS, sep='\t', index_col='mir')
    print(MIRSEQS)
    for row in MIRSEQS.iterrows():
        mirname = row[0].replace('*','_pass')
        site8 = row[1]['site8']

        all_seqs = generate_12mers(site8, False)
        print(len(all_seqs))
        temp = pd.DataFrame({'12mer': all_seqs, 'aligned_stype': [get_centered_stype(site8, seq) for seq in all_seqs]})

        with_site = temp[temp['aligned_stype'] != 'no site']
        with_site['bin'] = [x % 10 for x in np.arange(len(with_site))]
        for ix, group in with_site.groupby('bin'):
            group[['12mer', 'aligned_stype']].to_csv(os.path.join(options.OUTDIR, 'canon_{}_{}.txt'.format(mirname, ix)), sep='\t', index=False)

        no_site = temp[temp['aligned_stype'] == 'no site']
        no_site['bin'] = [x % 10 for x in np.arange(len(no_site))]
        for ix, group in no_site.groupby('bin'):
            group[['12mer', 'aligned_stype']].to_csv(os.path.join(options.OUTDIR, 'noncanon_{}_{}.txt'.format(mirname, ix)), sep='\t', index=False)

