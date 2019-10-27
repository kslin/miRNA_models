from concurrent.futures import ProcessPoolExecutor, as_completed
from optparse import OptionParser
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

import utils


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--sequence_file", dest="SEQUENCE_FILE", help="file with sequences")
    parser.add_option("--temp_folder", dest="TEMP_FOLDER", help="folder for writing temporary outputs")
    parser.add_option("--num_bg", dest="NUM_BG", type=int, help="number of background seqs to fold")
    parser.add_option("--num_processes", dest="NUM_PROCESSES", type=int, help="number of parallel processes")
    parser.add_option("--outfile", dest="OUTFILE", help="folder for writing outputs")

    (options, args) = parser.parse_args()

    if (not os.path.isdir(options.TEMP_FOLDER)):
        os.makedirs(options.TEMP_FOLDER)

    SEQUENCES = pd.read_csv(options.SEQUENCE_FILE, sep='\t', chunksize=options.NUM_PROCESSES)

    def get_RNAplfold(seq):
        cwd = os.getcwd()
        temp_subfolder = os.path.join(options.TEMP_FOLDER, seq)
        os.makedirs(temp_subfolder)
        os.chdir(temp_subfolder)

        # write sequence to a temporary file
        with open('temp.fa', 'w') as f:
            for ix in range(options.NUM_BG):
                bg = utils.generate_random_seq(14) + seq + utils.generate_random_seq(14)
                f.write('>{}\n{}\n'.format(ix, bg))

        # # call RNAplfold
        t0 = time.time()
        mycall = ['RNAplfold', '-L', '40', '-W', '40', '-u', '12']
        with open('temp.fa', 'r') as f:
            subprocess.call(mycall, shell=False, stdin=f, stdout=subprocess.PIPE)

        bg_vals = []
        for ix in range(options.NUM_BG):
            lunp_file = '{}_lunp'.format(ix)
            rnaplfold_data = pd.read_csv(lunp_file, sep='\t', header=1).set_index(' #i$').astype(float)
            bg_vals.append(rnaplfold_data.loc[26]['12'])

        os.chdir(cwd)
        shutil.rmtree(temp_subfolder)

        return bg_vals

    total = 0
    with open(options.OUTFILE, 'w') as outfile:
        for chunk in SEQUENCES:
            T0 = time.time()
            seqs = list(chunk['12mer'].values)
            total += len(seqs)
            print('Processing {}'.format(total))

            with ProcessPoolExecutor(max_workers=options.NUM_PROCESSES) as executor:
                results = executor.map(get_RNAplfold, seqs)

                for seq, result in zip(seqs, results):
                    for res in result:
                        outfile.write('{}\t{:.7f}\t{:.4f}\n'.format(seq, res, np.log(res)))

            print(time.time() - T0)

    shutil.rmtree(options.TEMP_FOLDER)

