import json
from optparse import OptionParser
import os

import numpy as np
import pandas as pd

import predict_helpers


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--features", dest="FEATURES", help="file with features")
    parser.add_option("--features_pass", dest="FEATURES_PASS", help="file with features for passenger strand, optional", default=None)
    parser.add_option("--model", dest="MODEL", help="json file with model parameters")
    parser.add_option("--freeAGO", dest="FREEAGO", help="value for free AGO concentration", type=float)
    parser.add_option("--freeAGO_pass", dest="FREEAGO_PASS", help="value for free AGO concentration of passenger strand, optional", type=float, default=None)
    parser.add_option("--kd_cutoff", dest="KD_CUTOFF", help="cutoff for KD values to keep", type=float, default=0.0)
    parser.add_option("--outfile", dest="OUTFILE", help="folder for writing outputs")

    (options, args) = parser.parse_args()

    # check that if passenger strand features are given, a passenger strand free AGO concentration is also given
    if options.FEATURES_PASS is not None:
        if options.FREEAGO_PASS is None:
            raise ValueError("Must give free AGO for passenger strand if passenger strand sequences are supplied.")

    # read in features
    features = predict_helpers.process_features(options.FEATURES, kd_cutoff=options.KD_CUTOFF)
    features['freeAGO'] = options.FREEAGO
    if options.FEATURES_PASS is not None:
        features_pass = predict_helpers.process_features(options.FEATURES_PASS, kd_cutoff=options.KD_CUTOFF)
        features_pass['freeAGO'] = options.FREEAGO_PASS
        features = pd.concat([features, features_pass], sort=False)

    # read in parameters
    with open(options.MODEL, 'r') as infile:
        TRAIN_PARAMS = json.load(infile)

    FITTED_PARAMS = {'log_decay': TRAIN_PARAMS['log_decay']}
    for feat, val in zip(TRAIN_PARAMS['FEATURE_LIST'][1:], TRAIN_PARAMS['feature_coefs']):
        FITTED_PARAMS[feat + '_coef'] = val

    for param in ['nosite_conc', 'utr3_coef', 'orf_coef']:
        if param in TRAIN_PARAMS:
            FITTED_PARAMS[param] = TRAIN_PARAMS[param]

    FEATURE_LIST = ','.join(TRAIN_PARAMS['FEATURE_LIST'][2:])

    _, predictions = predict_helpers.predict(features, FEATURE_LIST, FITTED_PARAMS)

    predictions.to_csv(options.OUTFILE, sep='\t')



