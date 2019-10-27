# miRNA_models

This repository contains models for predicting miRNA-mediated repression described in [our paper](https://www.biorxiv.org/content/10.1101/414763v1).

## Requirements
- python 3.6 or higher
- python packages listed in requirements.txt (swap out tensorflow-gpu with tensorflow if not using GPUs). To install, run `pip install -r requirements.txt`.
- RNAplfold from the [ViennaRNA package](https://www.tbi.univie.ac.at/RNA/)

## Modules

### cnn
This module contains code for building and training the combined CNN and biochemical models in train.py and monitoring the progress using Tensorboard. The parse_data_utils file contains helper functions for parsing tfrecords information and assembling it into the correct input format. The function models.seq2ka_predictor() builds the CNN for predicting KD from miRNA and target sequences.

Here is an example of how to predict relative KD values using the trained model from our paper:

	python cnn/generate_12mer_kds.py \
	--name all \
	--mirdata sample_data/inputs/mirseqs.txt \
	--mirlen 10 \
	--passenger \
	--load_model cnn/trained_model/model-100 \
	--outfile sample_data/outputs/kds/MIR_kds.txt


### rnaplfold
This module folds target sites in many different sequence contexts to calculate the basal accessibility of each site. We recommend using the only_canon flag to only calculate this value for canonical sites and avoid very long compute times.

First partition 12-nt kmers into 10 files:

	python rnaplfold/partition_seqs.py \
	--mirseqs sample_data/inputs/mirseqs.txt \
	--nbins 10 \
	--outdir sample_data/outputs/SA_background/sequences \
	--only_canon \
	--passenger

Then generate 200 random contexts for each sequence in each file and fold them. You may want to use a Makefile or snakemake to automate this step:

	for mirname in mir122_pass mir133_pass ; do \
	for ix in 0 1 2 3 4 5 6 7 8 9 ; do \
	bsub -R "rusage[mem=4096]" \
	python rnaplfold/get_SA_bg.py \
	--sequence_file sample_data/outputs/SA_background/sequences/canon_"$mirname"_"$ix".txt \
	--temp_folder sample_data/outputs/SA_background/bg_vals/canon_"$mirname"_"$ix"_TEMP \
	--num_bg 200 \
	--num_processes 24 \
	--outfile sample_data/outputs/SA_background/bg_vals/canon_"$mirname"_"$ix"_bg_vals.txt; \
	done; \
	done

Finally, parse RNAplfold outputs for all background sequences

	python rnaplfold/combine_results.py \
	--mirseqs sample_data/inputs/mirseqs.txt \
	--nbins 10 \
	--num_bg 200 \
	--infile_seqs sample_data/outputs/SA_background/sequences/canon_MIR_IX.txt \
	--infile_bg sample_data/outputs/SA_background/bg_vals/canon_MIR_IX_bg_vals.txt \
	--outfile sample_data/outputs/SA_background/bg_vals_processed/canon_MIR_bg_vals.txt \
	--passenger

The sequence and partitioned bg_vals can be deleted at this point.


### get_features
This module contains code that preprocesses data for both the biochemical model and the CNN. For best results, supply PCT scores.

First, navigate to temp folder and fold ORF + UTR3 sequences using RNAplfold

	mkdir sample_data/outputs/rnaplfold/TEMP
	cd sample_data/outputs/rnaplfold/TEMP
	RNAplfold -L 40 -W 80 -u 15 < ../../../../sample_data/inputs/orf_utr3.fa

Then, navigate back and process results for easier querying later

	python rnaplfold/process_mRNA_folding.py \
	--transcripts sample_data/inputs/transcripts.txt \
	--indir sample_data/outputs/rnaplfold/TEMP \
	--outdir sample_data/outputs/rnaplfold/rnaplfold_orf_utr3/

The TEMP files can be deleted at this point. To calculate all features:

	for mirname in mir122 mir133 mir122_pass mir133_pass ; do \
	python get_features/write_sites.py \
	--transcripts sample_data/inputs/transcripts.txt \
	--mir "$mirname" \
	--mirseqs sample_data/inputs/mirseqs.txt \
	--kds sample_data/outputs/kds/"$mirname"_kds.txt \
	--sa_bg sample_data/outputs/SA_background/bg_vals_processed/canon_"$mirname"_bg_vals.txt \
	--rnaplfold_dir sample_data/outputs/rnaplfold/rnaplfold_orf_utr3/ \
	--pct_file sample_data/inputs/pcts.txt \
	--overlap_dist 12 \
	--upstream_limit 15 \
	--outfile sample_data/outputs/features/"$mirname".txt ; \
	done


### biochem_model
This module contains code for building, training, and using the biochemical model and biochemical+ models.

Sample usage:

	python train.py \
	--tpm_file /lab/solexa_bartel/klin/miRNA_models_data/transfections/hek293ft_ensembl/processed/merged.txt \
	--mirseqs /lab/solexa_bartel/klin/miRNA_models_data/miRNA_info/HEK293FT_mirs/mirseqs.txt \
	--feature_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/$(current_model)/hek293ft/MIR.txt \
	--mode $${mode} \
	--init_bound \
	--passenger \
	--setparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)/train_$${mode}_with_passenger_MODEL_params.json \
	--kd_cutoff 0.0 \
	--extra_feats $${feats} \
	--outfile /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/fit_freeago_$${mode}_with_passenger_MODEL_preds.txt \
	--outparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/fit_freeago_$${mode}_with_passenger_MODEL_params.json
