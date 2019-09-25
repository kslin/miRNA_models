# miRNA_models

This repository contains models for predicting miRNA-mediated repression described in [our paper](https://www.biorxiv.org/content/10.1101/414763v1).

## Requirements
- python 3.6 or higher
- python packages listed in requirements.txt (swap out tensorflow-gpu with tensorflow if not using GPUs). To install, run `pip install -r requirements.txt`.
- RNAplfold from the [ViennaRNA package](https://www.tbi.univie.ac.at/RNA/)

## Modules

### rnaplfold
This module folds target sites in many different sequence contexts to calculate the basal accessibility of each site.

### get_features
This module contains code that preprocesses data for both the biochemical model and the CNN.

### biochem_model
This module contains code for building and training the biochemical model and biochemical+ models.

### cnn
This module contains code for building and training the combined CNN and biochemical models in train.py and monitoring the progress using Tensorboard. The parse_data_utils file contains helper functions for parsing tfrecords information and assembling it into the correct input format. The function models.seq2ka_predictor() builds the CNN for predicting KD from miRNA and target sequences.
