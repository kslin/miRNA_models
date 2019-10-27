from optparse import OptionParser
import os

import numpy as np
import pandas as pd


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--transcripts", dest="TRANSCRIPTS", help="file with transcript sequences")
    parser.add_option("--indir", dest="INDIR", help="folder with RNAplfold outputs")
    parser.add_option("--outdir", dest="OUTDIR", help="folder for writing outputs")

    (options, args) = parser.parse_args()

    transcripts = pd.read_csv(options.TRANSCRIPTS, sep='\t', index_col='transcript')

    for row in transcripts.iterrows():
        transcript_length = row[1]['orf_length'] + row[1]['utr3_length']

        # read in rnaplfold outputs
        lunp_file = os.path.join(options.INDIR, row[0]) + '_lunp'
        rnaplfold_data = pd.read_csv(lunp_file, sep='\t', header=1)
        rnaplfold_data = rnaplfold_data[rnaplfold_data.columns[:15]]

        # use parameters from Agarwal et al., 2015
        rnaplfold_data.columns = ['end'] + list(np.arange(14) + 1)
        rnaplfold_data = rnaplfold_data.set_index('end').astype(float)

        for ix in range(13):
            temp = rnaplfold_data.loc[ix+1]
            rnaplfold_data.loc[ix+1] = rnaplfold_data.loc[ix+1].fillna(temp[ix + 1])

        new_row = pd.Series({14: rnaplfold_data.iloc[-1][13]}, name=transcript_length+1)
        rnaplfold_data = rnaplfold_data[[14]].append(new_row)
        assert(len(rnaplfold_data) == len(rnaplfold_data.dropna()))

        # write to outfile
        outfile = os.path.join(options.OUTDIR, f'{row[0]}.txt')
        rnaplfold_data.to_csv(outfile, sep='\t')
