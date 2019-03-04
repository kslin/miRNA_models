import numpy as np
import pandas as pd

import utils


def calc_SPS(one_strand):
    deltaG_dict = {
        'AA': -0.93,
        'AT': -1.10,
        'AC': -2.24,
        'AG': -2.08,
        'TA': -1.33,
        'TT': -0.93,
        'TC': -2.35,
        'TG': -2.11,
        'CA': -2.11,
        'CT': -2.08,
        'CC': -3.26,
        'CG': -2.36,
        'GA': -2.35,
        'GT': -2.24,
        'GC': -3.42,
        'GG': -3.26,
    }

    if len(one_strand) < 2:
        raise ValueError('Length of sequence must be more than 1.')

    else:
        dG = 4.09  # initial energy
        ends = one_strand[0] + one_strand[1]
        dG += 0.45 * (int(one_strand[0] in ['A','T']) + int(one_strand[-1] in ['A','T']))  # terminal AU penalty

        if utils.rev_comp(one_strand) == one_strand:  # symmetry penalty
            dG += 0.43

        nts = 'X' + one_strand[0]
        for nt in one_strand[1:]:
            nts = nts[1:] + nt
            dG += deltaG_dict[nts]

    return dG

# print(calc_SPS('AAATTTA'))
# print(calc_SPS('AAATTT'))

from_vikram = pd.read_csv('../data/TA_SPS_by_seed_region.txt', sep='\t')
from_vikram['Seed region'] = [x.replace('U','T') for x in from_vikram['Seed region']]
from_vikram = from_vikram.set_index('Seed region')
dG_7merm8 = [calc_SPS(x) for x in from_vikram.index]
from_vikram['8mer'] = dG_7merm8
from_vikram['7mer-m8'] = dG_7merm8

dG_6mer = [calc_SPS(x[:-1]) for x in from_vikram.index]
from_vikram['7mer-a1'] = dG_6mer
from_vikram['6mer'] = dG_6mer

from_vikram['6mer-m8'] = [calc_SPS(x[1:]) for x in from_vikram.index]
from_vikram['6mer-a1'] = [calc_SPS(x[1:-1]) for x in from_vikram.index]
from_vikram['no site'] = np.nan

assert len(from_vikram[np.abs(from_vikram['SPS (8mer and 7mer-m8)'] - from_vikram['8mer']) > 0.001]) == 0
assert len(from_vikram[np.abs(from_vikram['SPS (7mer-1a and 6mer)'] - from_vikram['6mer']) > 0.001]) == 0

from_vikram[['8mer', '7mer-m8', '7mer-a1', '6mer', '6mer-m8', '6mer-a1', 'no site', 'TA']].to_csv('../data/TA_SPS_all.txt', sep='\t', float_format='%.3f')
