import operator
import re

import numpy as np
import pandas as pd

import utils


def _priority_order(locs, overlap_dist):
    """Helper function for get_sites_from_utr"""

    # make dictionary of loc occurences and order
    loc_dict = {}
    for ix, l in enumerate(locs):
        if l not in loc_dict:
            loc_dict[l] = (-1, ix)
        else:
            temp_count, temp_ix = loc_dict[l]
            loc_dict[l] = (temp_count - 1, temp_ix)

    loc_tuples = [(l, count, ix) for (l, (count, ix)) in loc_dict.items()]
    loc_tuples.sort(key=operator.itemgetter(1, 2))

    unique_locs = [t[0] for t in loc_tuples]
    nonoverlapping_locs = []
    prev = -100
    for l in unique_locs:
        if abs(l - prev) > overlap_dist:
            nonoverlapping_locs.append(l)
            prev = l

    return nonoverlapping_locs


def get_sites_from_utr(utr, site8, overlap_dist, only_canon):
    """
    Given a UTR sequence and the site sequence of an 8mer, return the location of all potential sites with >= 4nt of the 8mer.

    Parameters:
        utr (string): UTR sequence
        site (string): 8mer site sequence
        overlap_dist (int): overlap distance allowed between sites
        only_canon (bool): if true, only return location of canonical sites

    Returns:
        list of ints: location of all desired sites in a UTR
    """
    if only_canon:
        # get site locations of all canonical sites
        locs0 = [m.start() - 1 for m in re.finditer(site8[2:], utr)]  # pos 1-6
        locs1 = [m.start() for m in re.finditer(site8[1:-1], utr)]  # pos 2-7 (start of 6mer site)
        locs2 = [m.start() + 1 for m in re.finditer(site8[:-2], utr)]  # pos 3-8
        locs = (locs1 + locs2 + locs0)

    else:
        # get site locations of all 4mer subsequences of the 8mer site
        locs0 = [m.start() - 3 for m in re.finditer(site8[4:], utr)]  # pos 1-4
        locs1 = [m.start() - 2 for m in re.finditer(site8[3:-1], utr)]  # pos 2-5
        locs2 = [m.start() - 1 for m in re.finditer(site8[2:-2], utr)]  # pos 3-6
        locs3 = [m.start() for m in re.finditer(site8[1:-3], utr)]  # pos 4-7 (start of 6mer site)
        locs4 = [m.start() + 1 for m in re.finditer(site8[:-4], utr)]  # pos 5-8
        locs = (locs1 + locs2 + locs0 + locs3 + locs4)

    # get rid of any that would put the 6mer site outside the bounds of the UTR
    locs = [l for l in locs if ((l >= 0) and ((l + 6) <= len(utr)))]

    # if 1 or fewer sites found, return list as is
    if len(locs) > 1:
        locs = _priority_order(locs, overlap_dist)

    utr_ext = ('XXX' + utr + 'XXX')
    seqs = [utr_ext[l:l + 12] for l in locs]

    # sites already listed in order of priority, except longer sites take precedent
    return seqs, locs


### TS7 Feature functions ###
def calculate_threep_score(mirseq, utr, site_start, upstream_limit):
    """
    Calculate the three-prime pairing score

    Parameters
    ----------
    mirseq: string, miRNA sequence
    utr: string, utr sequence
    site_start: int, start of 12mer site
    upstream_limit: int, how far upstream to look for 3p pairing

    Output
    ------
    float: 3' pairing score
    """
    if site_start <= 0:
        return 0

    # get the 3' region of the mirna and the corresponding utr seq
    mirseq_3p = mirseq[8:]  # miRNA sequence from position 9 onward
    trailing = utr[max(0, site_start - upstream_limit): site_start + 2]  # site sequence up to edges of possible 8mer site
    utr_5p = utils.rev_comp(trailing)

    # initiate array for dynamic programming search
    scores = np.empty((len(utr_5p) + 1, len(mirseq_3p) + 1))
    scores.fill(np.nan)
    possible_scores = [0]

    # fill in array
    for i, nt1 in enumerate(utr_5p):
        for j, nt2 in enumerate(mirseq_3p):
            if nt1 == nt2:
                new_score = 0.5 + 0.5 * ((j > 3) & (j < 8))
                if not np.isnan(scores[i, j]):
                    new_score += scores[i, j]
                    scores[i + 1, j + 1] = new_score
                    possible_scores.append(new_score)
                else:
                    offset_penalty = max(0, (abs(i - j) - 2) * 0.5)
                    scores[i + 1, j + 1] = new_score - offset_penalty
            else:
                scores[i + 1, j + 1] = float('NaN')

    return np.nanmax(possible_scores)


def calculate_local_au(utr, site_start):
    """
    Calculate the local AU score

    Parameters
    ----------
    utr: string, utr sequence
    site_start: int, start of 12mer site

    Output
    ------
    float: local AU score
    """
    # find A, U and weights upstream of site
    upstream = utr[max(0, site_start - 30): max(0, site_start)]
    upstream = [int(x in ['A', 'U']) for x in upstream]
    upweights = [1.0 / (x + 1) for x in range(len(upstream))][::-1]

    # find A,U and weights downstream of site
    downstream = utr[site_start + 12:min(len(utr), site_start + 42)]
    downstream = [int(x in ['A', 'U']) for x in downstream]
    downweights = [1.0 / (x + 1) for x in range(len(downstream))]

    weighted = np.dot(upstream, upweights) + np.dot(downstream, downweights)
    total = float(sum(upweights) + sum(downweights))

    return weighted / total


def get_ts7_features(mirseq, locs, stypes, utr, utr_len, orf_len, upstream_limit, rnaplfold_data, pct_df):
    # calculate TS7 features
    features = []
    for loc, stype in zip(locs, stypes):

        # get ts7 features
        local_au = calculate_local_au(utr, loc - 3)
        threep = calculate_threep_score(mirseq, utr, loc - 3, upstream_limit)
        min_dist = min(loc, utr_len - (loc + 6))
        assert (min_dist >= 0), (loc, utr_len)

        # use the rnaplfold data to calculate the site accessibility
        if rnaplfold_data is None:
            sa_score = 0
        else:
            site_start_for_SA = loc + 7
            if (site_start_for_SA) not in rnaplfold_data.index:
                sa_score = 0
            else:
                row_vals = rnaplfold_data.loc[site_start_for_SA].values[:14]  # pos 1-14 unpaired, Agarwal 2015
                # row_vals = rnaplfold_data.loc[site_start_for_SA].values[:10]  # pos 1-10 unpaired, Sean

                for raw_sa_score in row_vals[::-1]:
                    if not np.isnan(raw_sa_score):
                        break

                if np.isnan(raw_sa_score):
                    sa_score = np.nan
                elif raw_sa_score <= 0:
                    sa_score = -5.0
                    print("warning, nan sa_score")
                else:
                    sa_score = np.log10(raw_sa_score)

        # get PCT
        try:
            if stype in ['6mer', '7mer-a1', '7mer-m8', '8mer']:
                pct = pct_df.loc[loc]['PCT']
            else:
                pct = 0.0

        except:
            print(pct_df)
            raise ValueError('PCT locations do not match for {}'.format(transcript))

        features.append([local_au, threep, sa_score, min_dist/2000.0, utr_len/2000.0, orf_len/2000.0, pct])

    features = pd.DataFrame(np.array(features).astype(float))
    features.columns = ['Local_AU', 'Threep', 'SA', 'Min_dist', 'UTR_len', 'ORF_len', 'PCT']

    return features
