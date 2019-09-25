current_model="feat1_allgenes_lr003_nodropout_batch50_rebalancekds2k_noaugment_repweight095_mask_w3_netpred"

train_drop_one_mir:
	for mirname in lsy6 mir1 mir124 mir137 mir139 mir143 mir144 mir153 mir155 mir182 mir199a mir204 mir205 mir216b mir223 mir7 ; do \
		for feats in "none" "logSA_diff,Threep_canon,PCT" ; do \
			bsub python train.py \
				--tpm_file /lab/solexa_bartel/klin/miRNA_models_data/transfections/charlie17_refseq/processed/merged.txt \
				--mirseqs /lab/solexa_bartel/klin/miRNA_models_data/miRNA_info/charlie_mirs/mirseqs_no_let7_drop_$${mirname}.txt \
				--feature_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/$(current_model)/hela/MIR.txt \
				--mode all \
				--passenger \
				--init_bound \
				--kd_cutoff 0.0 \
				--extra_feats $${feats} \
				--outfile /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)/train_all_with_passenger_MODEL_drop_$${mirname}_preds.txt \
				--outparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)/train_all_with_passenger_MODEL_drop_$${mirname}_params.json ; \
		done ; \
	done

train_only_with_passenger:
	mkdir -p /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)
	for mode in all canon noncanon ; do \
		for feats in "none" "logSA_diff,Threep_canon,PCT" ; do \
			bsub python train.py \
				--tpm_file /lab/solexa_bartel/klin/miRNA_models_data/transfections/charlie17_refseq/processed/merged.txt \
				--mirseqs ../data/mirseqs_no_let7.txt \
				--feature_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/$(current_model)/hela/MIR.txt \
				--mode $${mode} \
				--passenger \
				--init_bound \
				--kd_cutoff 0.0 \
				--extra_feats $${feats} \
				--outfile /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)/train_$${mode}_with_passenger_MODEL_preds.txt \
				--outparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)/train_$${mode}_with_passenger_MODEL_params.json ; \
		done ; \
	done

train_only_no_passenger:
	mkdir -p /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)
	for mode in all canon noncanon ; do \
		for feats in "none" "logSA_diff,Threep_canon,PCT" ; do \
			bsub python train.py \
				--tpm_file /lab/solexa_bartel/klin/miRNA_models_data/transfections/charlie17_refseq/processed/merged.txt \
				--mirseqs ../data/mirseqs_no_let7.txt \
				--feature_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/$(current_model)/hela/MIR.txt \
				--mode $${mode} \
				--init_bound \
				--kd_cutoff 0.0 \
				--extra_feats $${feats} \
				--outfile /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)/train_$${mode}_no_passenger_MODEL_preds.txt \
				--outparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)/train_$${mode}_no_passenger_MODEL_params.json ; \
		done ; \
	done

test_fit_freeAGO_with_passenger:
	mkdir -p /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)
	for mode in all canon noncanon ; do \
		for feats in "none" "logSA_diff,Threep_canon,PCT" ; do \
			bsub python train.py \
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
				--outparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/fit_freeago_$${mode}_with_passenger_MODEL_params.json ; \
		done ; \
	done

test_fit_freeAGO_no_passenger:
	mkdir -p /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)
	for mode in all canon noncanon ; do \
		for feats in "none" "logSA_diff,Threep_canon,PCT" ; do \
			bsub python train.py \
				--tpm_file /lab/solexa_bartel/klin/miRNA_models_data/transfections/hek293ft_ensembl/processed/merged.txt \
				--mirseqs /lab/solexa_bartel/klin/miRNA_models_data/miRNA_info/HEK293FT_mirs/mirseqs.txt \
				--feature_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/$(current_model)/hek293ft/MIR.txt \
				--mode $${mode} \
				--init_bound \
				--setparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)/train_$${mode}_no_passenger_MODEL_params.json \
				--kd_cutoff 0.0 \
				--extra_feats $${feats} \
				--outfile /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/fit_freeago_$${mode}_no_passenger_MODEL_preds.txt \
				--outparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/fit_freeago_$${mode}_no_passenger_MODEL_params.json ; \
		done ; \
	done

# mir184 mir145 mir133 mir122 mir217 mir216a mir219a mir451a mir375 mir138 mir200b mir190a
test_fit_freeAGO_with_passenger_shuffle_noncanon:
	mkdir -p /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/shuffles
	mkdir -p /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/lsf_outputs
	for mode in all noncanon; do \
		for feats in "none" "logSA_diff,Threep_canon,PCT" ; do \
			for mirname in mir184 mir145 mir133 mir122 mir217 mir216a mir219a mir451a mir375 mir138 mir200b mir190a ; do \
				for shuffle_mir in mir184 mir145 mir133 mir122 mir217 mir216a mir219a mir451a mir375 mir138 mir200b mir190a ; do \
					bsub -o /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/lsf_outputs/$${mode}_$${feats}_$${mirname}_$${shuffle_mir}.txt \
					python train.py \
						--tpm_file /lab/solexa_bartel/klin/miRNA_models_data/transfections/hek293ft_ensembl/processed/merged.txt \
						--mirseqs /lab/solexa_bartel/klin/miRNA_models_data/miRNA_info/HEK293FT_mirs/mirseqs.txt \
						--feature_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/$(current_model)/hek293ft/MIR.txt \
						--mir_to_shuffle $${mirname} \
						--shuffle_mir $${shuffle_mir} \
						--shuffle_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/$(current_model)/hek293ft_shuffled/MIR_SHUFFLE.txt \
						--mode $${mode} \
						--init_bound \
						--passenger \
						--setparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)/train_$${mode}_with_passenger_MODEL_params.json \
						--kd_cutoff 0.0 \
						--extra_feats $${feats} \
						--outfile /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/shuffles/fit_freeago_$${mode}_with_passenger_MODEL_shuffle_$${mirname}_$${shuffle_mir}_preds.txt \
						--outparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/shuffles/fit_freeago_$${mode}_with_passenger_MODEL_shuffle_$${mirname}_$${shuffle_mir}_params.json ; \
				done ; \
			done ; \
		done ; \
	done

test_fit_freeAGO_no_passenger_shuffle_noncanon:
	mkdir -p /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/shuffles
	mkdir -p /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/lsf_outputs
	for mode in all noncanon; do \
		for feats in "none" "logSA_diff,Threep_canon,PCT" ; do \
			for mirname in mir184 mir145 mir133 mir122 mir217 mir216a mir219a mir451a mir375 mir138 mir200b mir190a ; do \
				for shuffle_mir in mir184 mir145 mir133 mir122 mir217 mir216a mir219a mir451a mir375 mir138 mir200b mir190a ; do \
					bsub -o /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/lsf_outputs/$${mode}_$${feats}_$${mirname}_$${shuffle_mir}.txt \
					python train.py \
						--tpm_file /lab/solexa_bartel/klin/miRNA_models_data/transfections/hek293ft_ensembl/processed/merged.txt \
						--mirseqs /lab/solexa_bartel/klin/miRNA_models_data/miRNA_info/HEK293FT_mirs/mirseqs.txt \
						--feature_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/$(current_model)/hek293ft/MIR.txt \
						--mir_to_shuffle $${mirname} \
						--shuffle_mir $${shuffle_mir} \
						--shuffle_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/$(current_model)/hek293ft_shuffled/MIR_SHUFFLE.txt \
						--mode $${mode} \
						--init_bound \
						--setparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hela_transfection/$(current_model)/train_$${mode}_no_passenger_MODEL_params.json \
						--kd_cutoff 0.0 \
						--extra_feats $${feats} \
						--outfile /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/shuffles/fit_freeago_$${mode}_no_passenger_MODEL_shuffle_$${mirname}_$${shuffle_mir}_preds.txt \
						--outparams /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/$(current_model)/shuffles/fit_freeago_$${mode}_no_passenger_MODEL_shuffle_$${mirname}_$${shuffle_mir}_params.json ; \
				done ; \
			done ; \
		done ; \
	done
