from optparse import OptionParser
import os

import numpy as np
import pandas as pd

import utils


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--mirseqs", dest="MIRSEQS", help="table with miRNA sequence information")
    parser.add_option("--nbins", dest="NBINS", type=int, help="number of files to partition into")
    parser.add_option("--outdir", dest="OUTDIR", help="folder for writing outputs")
    parser.add_option("--only_canon", dest="ONLY_CANON", help="only calculate for canonical sites", default=False, action='store_true')
    parser.add_option("--passenger", dest="PASSENGER", help="also calculate for passenger strand", default=False, action='store_true')

    (options, args) = parser.parse_args()

    if (not os.path.isdir(options.OUTDIR)):
        os.makedirs(options.OUTDIR)

    MIRSEQS = pd.read_csv(options.MIRSEQS, sep='\t', index_col='mir')

    for row in MIRSEQS.iterrows():
        # get names and 8mer sites for the guide and passenger strands, if applicable
        mirnames = [row[0]]
        site8s = [utils.rev_comp(row[1]['guide_seq'][1:8]) + 'A']
        if options.PASSENGER:
            mirnames += [row[0] + '_pass']
            site8s += [utils.rev_comp(row[1]['pass_seq'][1:8]) + 'A']
        
        for mirname, site8 in zip(mirnames, site8s):
            all_seqs = utils.generate_12mers(site8, options.ONLY_CANON)
            print(len(all_seqs))
            temp = pd.DataFrame({'12mer': all_seqs, 'aligned_stype': [utils.get_centered_stype(site8, seq) for seq in all_seqs]})

            # separate by canonical or not canonical and partition into files
            with_site = temp[temp['aligned_stype'] != 'no site']
            with_site['bin'] = [x % options.NBINS for x in np.arange(len(with_site))]
            for ix, group in with_site.groupby('bin'):
                group[['12mer', 'aligned_stype']].to_csv(os.path.join(options.OUTDIR, 'canon_{}_{}.txt'.format(mirname, ix)), sep='\t', index=False)

            if not options.ONLY_CANON:
                no_site = temp[temp['aligned_stype'] == 'no site']
                no_site['bin'] = [x % options.NBINS for x in np.arange(len(no_site))]
                for ix, group in no_site.groupby('bin'):
                    group[['12mer', 'aligned_stype']].to_csv(os.path.join(options.OUTDIR, 'noncanon_{}_{}.txt'.format(mirname, ix)), sep='\t', index=False)

