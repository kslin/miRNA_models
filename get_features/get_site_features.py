import operator
import re

import numpy as np
import pandas as pd

import utils


def get_sites_from_kd_dict(transcript_id, sequence, kd_dict, overlap_dist):
    if len(sequence) < 9:
        return pd.DataFrame(None)

    mir_info = {
        'prev_loc': -100,
        'prev_seq': '',
        'prev_kd': 100,
        'keep_kds': [],
        'keep_locs': [],
        'keep_seqs': []
    }

    pad_seq = 'XXX' + sequence + 'XXX'
    seq = 'A' + pad_seq[:11]

    # iterate through 12mers in the sequence
    for loc, nt in enumerate(pad_seq[11:]):
        seq = seq[1:] + nt

        if seq in kd_dict:
            new_kd = kd_dict[seq]

            # if new site is too close to previous site, take site with higher affinity
            if (loc - mir_info['prev_loc']) <= overlap_dist:
                if new_kd < mir_info['prev_kd']:
                    mir_info['keep_kds'][-1] = new_kd
                    mir_info['keep_locs'][-1] = loc
                    mir_info['keep_seqs'][-1] = seq
                    mir_info['prev_loc'] = loc
                    mir_info['prev_kd'] = new_kd
                    # print('replace')
                else:
                    # print('skipped')
                    continue
            else:
                # print('added')
                mir_info['keep_kds'].append(new_kd)
                mir_info['keep_locs'].append(loc)
                mir_info['keep_seqs'].append(seq)
                mir_info['prev_loc'] = loc
                mir_info['prev_kd'] = new_kd
                
    all_sites = pd.DataFrame({
        'transcript': transcript_id,
        '12mer': mir_info['keep_seqs'],
        'log_kd': mir_info['keep_kds'],
        'loc': mir_info['keep_locs']
    })

    return all_sites


def get_sites_from_kd_dict_improved(transcript_id, sequence, kd_dict, overlap_dist):
    if len(sequence) < 9:
        return pd.DataFrame(None)

    pad_seq = 'XXX' + sequence + 'XXX'
    seq = 'A' + pad_seq[:11]

    all_sites = []

    # iterate through 12mers in the sequence
    for loc, nt in enumerate(pad_seq[11:]):
        seq = seq[1:] + nt

        if seq in kd_dict:
            new_kd = kd_dict[seq]
            all_sites.append([seq, new_kd, loc])

    if len(all_sites) == 0:
        return pd.DataFrame(None)
                
    all_sites = pd.DataFrame(all_sites, columns=['12mer','log_kd','loc']).sort_values('log_kd')

    all_locs = all_sites['loc'].values
    keep_locs = [all_locs[0]]


    for loc in all_locs[1:]:
        if np.min([np.abs(x - loc) for x in keep_locs]) > overlap_dist:
            keep_locs.append(loc)

    all_sites = all_sites[all_sites['loc'].isin(keep_locs)]
    all_sites['transcript'] = transcript_id

    return all_sites.sort_values('loc')


def _priority_order(locs, overlap_dist):
    """Helper function for get_sites_from_utr"""
    
    temp = pd.DataFrame({'loc': locs})
    temp = temp.drop_duplicates(keep='first')
    temp['priority'] = np.arange(len(temp))

    new_locs = []
    while len(temp) > 0:
        top_loc = temp.iloc[0]['loc']
        new_locs.append(top_loc)
        temp = temp[np.abs(temp['loc'] - top_loc) > overlap_dist].sort_values('priority')

    return sorted(new_locs)


def get_sites_from_sequence(transcript_id, sequence, site8, overlap_dist, only_canon):
    """
    Given a nucleotide sequence and the site sequence of an 8mer, return the location of all potential sites with >= 4nt of the 8mer.

    Parameters:
        sequence (string): nucleotide sequence
        site (string): 8mer site sequence
        overlap_dist (int): overlap distance allowed between sites
        only_canon (bool): if true, only return location of canonical sites

    Returns:
        list of ints: location of all desired sites in a sequence
    """
    # get site locations of all canonical sites
    locs0 = [m.start() - 1 for m in re.finditer('(?={})'.format(site8[2:]), sequence)]  # pos 1-6
    locs1 = [m.start() for m in re.finditer('(?={})'.format(site8[1:-1]), sequence)]  # pos 2-7 (start of 6mer site)
    locs2 = [m.start() + 1 for m in re.finditer('(?={})'.format(site8[:-2]), sequence)]  # pos 3-8
    locs = (locs1 + locs2 + locs0)

    if not only_canon:
        # get site locations of all 4mer subsequences of the 8mer site
        locs0 = [m.start() - 3 for m in re.finditer('(?={})'.format(site8[4:]), sequence)]  # pos 1-4
        locs1 = [m.start() - 2 for m in re.finditer('(?={})'.format(site8[3:-1]), sequence)]  # pos 2-5
        locs2 = [m.start() - 1 for m in re.finditer('(?={})'.format(site8[2:-2]), sequence)]  # pos 3-6
        locs3 = [m.start() for m in re.finditer('(?={})'.format(site8[1:-3]), sequence)]  # pos 4-7 (start of 6mer site)
        locs4 = [m.start() + 1 for m in re.finditer('(?={})'.format(site8[:-4]), sequence)]  # pos 5-8
        locs += (locs1 + locs2 + locs0 + locs3 + locs4)

    # get rid of any that would put the 6mer site outside the bounds of the sequence
    locs = [l for l in locs if ((l >= 0) and ((l + 6) <= len(sequence)))]

    # if 1 or fewer sites found, return list as is
    if len(locs) > 1:
        locs = _priority_order(locs, overlap_dist)

    sequence_ext = ('XXX' + sequence + 'XXX')
    seqs = [sequence_ext[l:l + 12] for l in locs]

    all_sites = pd.DataFrame({
        'transcript': transcript_id,
        '12mer': seqs,
        'loc': locs
    })

    return all_sites


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
    upstream = [int(x in ['A', 'T']) for x in upstream]
    upweights = [1.0 / (x + 1) for x in range(len(upstream))][::-1]

    # find A,U and weights downstream of site
    downstream = utr[site_start + 12:min(len(utr), site_start + 42)]
    downstream = [int(x in ['A', 'T']) for x in downstream]
    downweights = [1.0 / (x + 1) for x in range(len(downstream))]

    weighted = np.dot(upstream, upweights) + np.dot(downstream, downweights)
    total = float(sum(upweights) + sum(downweights))

    return weighted / total


def get_ts7_features(mirseq, locs, stypes, sequence, utr_len, orf_len, upstream_limit, rnaplfold_data, pct_df):
    # calculate TS7 features
    features = []
    for loc, stype in zip(locs, stypes):
        in_orf = loc < orf_len

        # get ts7 features
        local_au = calculate_local_au(sequence, loc - 3)
        threep = calculate_threep_score(mirseq, sequence, loc - 3, upstream_limit)
        min_dist = min(abs(loc - orf_len), abs(loc - utr_len))
        # # min_dist = min(loc, utr_len - (loc + 6))
        # if in_orf:
        #     min_dist = (orf_len - (loc + 6)) + utr_len
        # else:
        #     min_dist = utr_len - (loc + 6)

        # assert (min_dist >= 0), (loc, utr_len, orf_len, in_orf)

        # use the rnaplfold data to calculate the site accessibility
        if rnaplfold_data is None:
            sa_score = None
        else:
            site_start_for_SA = loc + 6
            if (site_start_for_SA) not in rnaplfold_data.index:
                sa_score = None
            else:
                sa_score = np.log(rnaplfold_data.loc[site_start_for_SA]['15'])  # pos 2-16 unpaired, biochem model
                if np.isnan(sa_score):
                    sa_score = None

                # row_vals = rnaplfold_data.loc[site_start_for_SA].values[:14]  # pos 1-14 unpaired, Agarwal 2015
                # row_vals = rnaplfold_data.loc[site_start_for_SA].values[:10]  # pos 1-10 unpaired, Sean

                # for raw_sa_score in row_vals[::-1]:
                #     if not np.isnan(raw_sa_score):
                #         break

                # if np.isnan(raw_sa_score):
                #     sa_score = np.nan
                # elif raw_sa_score <= 0:
                #     sa_score = -5.0
                #     print("warning, nan sa_score")
                # else:
                #     sa_score = np.log10(raw_sa_score)

        # get PCT
        pct = 0.0
        if (not in_orf) & (pct_df is not None):
            try:
                if stype in ['6mer', '7mer-a1', '7mer-m8', '8mer']:
                    pct = pct_df.loc[loc - orf_len]['PCT']

            except:
                print(loc, orf_len)
                print(pct_df)
                raise ValueError('PCT locations do not match')

        # features.append([float(in_orf), min_dist/2000.0, local_au, threep, sa_score, pct, utr_len/2000.0, orf_len/2000.0])
        features.append([float(in_orf), sa_score, threep, pct, local_au, min_dist/2000.0, utr_len/2000.0, orf_len/2000.0])

    return np.array(features).astype(float)
