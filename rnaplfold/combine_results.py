from optparse import OptionParser
import os

import numpy as np
import pandas as pd

import utils


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--mirseqs", dest="MIRSEQS", help="miRNA info")
    parser.add_option("--nbins", dest="NBINS", type=int, help="number of files to combine")
    parser.add_option("--num_bg", dest="NUM_BG", type=int, help="number of background sequences to expect")
    parser.add_option("--infile_seqs", dest="INFILE_SEQS", help="files with 12mer sequences, use MIR and IX as placeholders")
    parser.add_option("--infile_bg", dest="INFILE_BG", help="files with rnaplfold outputs, use MIR and IX as placeholders")
    parser.add_option("--outfile", dest="OUTFILE", help="file for writing outputs, use MIR as placeholder")
    parser.add_option("--passenger", dest="PASSENGER", help="also calculate for passenger strand", default=False, action='store_true')

    (options, args) = parser.parse_args()

    mirseqs = pd.read_csv(options.MIRSEQS, sep='\t', index_col='mir')

    for row in mirseqs.iterrows():
        # get names and 8mer sites for the guide and passenger strands, if applicable
        mirnames = [row[0]]
        site8s = [utils.rev_comp(row[1]['guide_seq'][1:8]) + 'A']
        if options.PASSENGER:
            mirnames += [row[0] + '_pass']
            site8s += [utils.rev_comp(row[1]['pass_seq'][1:8]) + 'A']

        # read in RNAplfold results
        for mirname, site8 in zip(mirnames, site8s):
            SA_bg = []
            for ix in range(options.NBINS):
                seqs = pd.read_csv(os.path.join(options.INFILE_SEQS.replace('MIR', mirname).replace('IX', str(ix))), sep='\t')
                temp = pd.read_csv(os.path.join(options.INFILE_BG.replace('MIR', mirname).replace('IX', str(ix))), sep='\t', header=None)
                temp.columns = ['12mer','p','logp']
                temp['count'] = 1
                temp = temp.groupby('12mer').agg({'p': np.mean, 'logp': np.mean, 'count': np.sum})
                if len(temp[temp['count'] != options.NUM_BG]) > 0:
                    print(mirname, ix)
                    raise ValueError(f'expected {options.NUM_BG} background sequences')
                if len(temp) != len(seqs):
                    print(mirname, ix, len(temp), len(seqs))
                    raise ValueError('not all seqs')
                
                SA_bg.append(temp.drop('count', 1))
            SA_bg = pd.concat(SA_bg).reset_index()
            SA_bg_X = [SA_bg]

            # add 12mer sequences for edge sequences
            for ix in range(3):
                temp1 = SA_bg.copy()
                temp1['12mer'] = [('X' * (ix + 1)) + x[ix+1:] for x in temp1['12mer']]
                temp1 = temp1.groupby(['12mer']).agg(np.mean).reset_index()

                temp2 = SA_bg.copy()
                temp2['12mer'] = [x[:-(ix+1)] + ('X' * (ix + 1)) for x in temp2['12mer']]
                temp2 = temp2.groupby(['12mer']).agg(np.mean).reset_index()
                SA_bg_X += [temp1, temp2]
        
            SA_bg_X = pd.concat(SA_bg_X)
            SA_bg_X['mir'] = mirname
            SA_bg_X['stype'] = [utils.get_centered_stype(site8, x) for x in SA_bg_X['12mer'].values]
            print(mirname, len(SA_bg_X[SA_bg_X['stype'] != 'no site']))
            SA_bg_X.to_csv(os.path.join(options.OUTFILE.replace('MIR', mirname)), sep='\t', index=False, float_format='%.4f')
