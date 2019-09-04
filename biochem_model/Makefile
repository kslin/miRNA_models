predictions_5mirs:
	python predict.py \
		--transcripts /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/resubmission/transcripts_orf90.txt \
		--kds /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/kds.txt \
		--sa_bg /lab/solexa_bartel/klin/RNAplfold/RNAplfold_bg200/SA_bg/canon_MIR_bg_vals.txt \
		--pct_file /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_output_files/refseq_utr90_fixed/features.txt \
		--mirs lsy6,mir1,mir124,mir155,mir7 \
		--rnaplfold_dir /lab/solexa_bartel/klin/RNAplfold/RNAplfold_orf_utr3_u25_processed/ \
		--kd_cutoff "10" \
		--outfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/resubmission/features_SA_PCT.txt


train_only:
	python train.py \
		--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
		--mirseqs ../data/mirseqs_no_let7.txt \
		--feature_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/MIR.txt \
		--mode train_only \
		--extra_feats "logSA_diff,Threep_canon,PCT" \
		--passenger


test_only:
	mkdir -p /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection
	python train.py \
		--tpm_file /lab/solexa_bartel/klin/miRNA_models_data/transfections/hek293ft_ensembl/processed/merged.txt \
		--mirseqs /lab/solexa_bartel/klin/miRNA_models_data/miRNA_info/HEK293FT_mirs/mirseqs.txt \
		--feature_file /lab/solexa_bartel/klin/miRNA_models_data/model_inputs/biochem/predicted_kds/MIR.txt \
		--mode test_only \
		--extra_feats "logSA_diff,Threep_canon,PCT" \
		--outfile /lab/solexa_bartel/klin/miRNA_models_data/model_outputs/biochem/hek293ft_transfection/fit_freeago_no_passenger.txt


train_cnn_test:
	for mirname in mir137 ; do \
		python train.py \
			--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
			--mirseqs ../data/mirseqs_no_let7.txt \
			--feature_file /lab/solexa_bartel/klin/CNN/feature_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${mirname}/MIR.txt \
			--test_mir $${mirname} \
			--mode shuffle_noncanon \
			--extra_feats "logSA_diff,Threep_canon,PCT" \
			--passenger ; \
	done


train_cnn_test_all:
	mkdir -p /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/all_test_all && \
	for mirname in mir137 mir139 mir143 mir144 mir153 mir182 mir199a mir204 mir205 mir216b mir223 ; do \
		bsub -R "rusage[mem=4096]" \
		-m "it-c06b03 it-c06b04 it-c06b06 it-c06b07 it-c06b08" \
		-o /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/lsf_outputs/all_test_all_$${mirname}.out \
			python train.py \
				--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
				--mirseqs ../data/mirseqs_no_let7.txt \
				--feature_file /lab/solexa_bartel/klin/CNN/feature_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${mirname}/MIR.txt \
				--test_mir $${mirname} \
				--mode all \
				--extra_feats "logSA_diff,Threep_canon,PCT" \
				--outfile /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/all_test_all/$${mirname}.txt \
				--passenger ; \
	done


train_cnn_modes:
	for mode in all canon noncanon ; do \
		mkdir -p /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${mode} && \
		for mirname in mir137 mir139 mir143 mir144 mir153 mir182 mir199a mir204 mir205 mir216b mir223 ; do \
			bsub -R "rusage[mem=4096]" \
				-m "it-c06b03 it-c06b04 it-c06b06 it-c06b07 it-c06b08 it-c06b10 it-c06b11 it-c06b12 it-c06b14" \
				-o /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/lsf_outputs/$${mode}_$${mirname}.out \
				python train.py \
					--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
					--mirseqs ../data/mirseqs_no_let7.txt \
					--feature_file /lab/solexa_bartel/klin/CNN/feature_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${mirname}/MIR.txt \
					--test_mir $${mirname} \
					--mode $${mode} \
					--extra_feats "logSA_diff,Threep_canon,PCT" \
					--outfile /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${mode}/$${mirname}.txt \
					--passenger ; \
		done ; \
	done

# none logSA_diff Threep_canon PCT
train_cnn_step_feats:
	for feat in none logSA_diff Threep_canon PCT ; do \
		mkdir -p /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${feat} && \
		for mirname in mir137 mir139 mir143 mir144 mir153 mir182 mir199a mir204 mir205 mir216b mir223 ; do \
			bsub -R "rusage[mem=4096]" \
				-m "it-c06b03 it-c06b04 it-c06b06 it-c06b07 it-c06b08 it-c06b10 it-c06b11 it-c06b12 it-c06b14" \
				-o /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/lsf_outputs/$${feat}_$${mirname}.out \
				python train.py \
					--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
					--mirseqs ../data/mirseqs_no_let7.txt \
					--feature_file /lab/solexa_bartel/klin/CNN/feature_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${mirname}/MIR.txt \
					--test_mir $${mirname} \
					--mode all \
					--extra_feats $${feat} \
					--outfile /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${feat}/$${mirname}.txt \
					--passenger ; \
		done ; \
	done


train_cnn_shuffles:
	for mode in shuffle_noncanon only_shuffle_noncanon ; do \
		mkdir -p /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${mode} && \
		for mirname in mir137 mir139 mir143 mir144 mir153 mir182 mir199a mir204 mir205 mir216b mir223 ; do \
			bsub -R "rusage[mem=4096]" \
				-m "it-c06b02 it-c06b03 it-c06b04 it-c06b05 it-c06b06 it-c06b07 it-c06b08 it-c06b12 it-c06b14" \
				-o /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/lsf_outputs/$${mode}_$${mirname}.out \
				python train.py \
					--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
					--mirseqs ../data/mirseqs_no_let7.txt \
					--feature_file /lab/solexa_bartel/klin/CNN/feature_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${mirname}/MIR.txt \
					--test_mir $${mirname} \
					--mode $${mode} \
					--extra_feats "logSA_diff,Threep_canon,PCT" \
					--outfile /lab/solexa_bartel/klin/CNN/repression_predictions/occupancy_only_final_xval_batch3_withclipping_dropout0_withreg_netpred2_round2/$${mode}/$${mirname}_REPLACE.txt \
					--passenger ; \
		done ; \
	done


predictions_mir155_mouse:
	python predict.py \
		--transcripts /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/resubmission/mouse/transcripts.txt \
		--kds /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/kds.txt \
		--sa_bg /lab/solexa_bartel/klin/RNAplfold/RNAplfold_bg200/SA_bg/canon_bg_vals.txt \
		--mirs mir155 \
		--rnaplfold_dir /lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/RNAplfold_orf_utr3_u15_mouse/ \
		--kd_cutoff 0 \
		--outfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/resubmission/mouse/features.txt

features_test:
	python write_features.py \
		--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged_mini.txt \
		--kd_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/kds.txt \
		--orf_file /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_input_files/refseq_utr90_fixed/orf90.txt \
		--mirseqs mirseqs.txt \
		--pct /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_output_files/refseq_utr90_fixed/features.txt \
		--ta_sps TA_SPS_all.txt \
		--rnaplfold_folder /lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/RNAplfold_utr90_fixed_u15 \
		--outdir outputs/test \
		--overlap_dist 7 \
		--upstream_limit 15 \
		--only_canon \
		--only_rbns

features_all_sites_5mirs:
	python write_features.py \
		--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
		--kd_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/kds.txt \
		--orf_file /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_input_files/refseq_utr90_fixed/orf90.txt \
		--mirseqs mirseqs.txt \
		--pct /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_output_files/refseq_utr90_fixed/features.txt \
		--ta_sps TA_SPS_all.txt \
		--rnaplfold_folder /lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/RNAplfold_utr90_fixed_u15 \
		--outdir outputs/all_sites_5mirs \
		--overlap_dist 7 \
		--upstream_limit 15 \
		--only_rbns

features_only_canon:
	python write_features.py \
		--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
		--kd_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/kds.txt \
		--orf_file /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_input_files/refseq_utr90_fixed/orf90.txt \
		--mirseqs mirseqs.txt \
		--pct /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_output_files/refseq_utr90_fixed/features.txt \
		--ta_sps TA_SPS_all.txt \
		--rnaplfold_folder /lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/RNAplfold_utr90_fixed_u15 \
		--outdir outputs/only_canon \
		--overlap_dist 7 \
		--upstream_limit 15 \
		--only_canon

TS7_models:
	for model_type in TS7 \
					  TS7_multisite ; do \
		bsub -q 18 -o outputs/model_results/only_canon_16mirs/$${model_type}.txt \
			python train.py \
			--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
			--feature_file outputs/only_canon/features.txt \
			--out_folder outputs/model_results/only_canon_16mirs \
			--model_type $${model_type} ; \
	done

only_canon_linear_models:
	for model_type in TS7_multisite_5mirs \
					  TS7_multisite_5mirs_logKA \
				      TS7_multisite_5mirs_logKA_noredundant \
					  TS7_multisite_5mirs_logKA_noredundant_6canon ; do \
		bsub -q 18 -o outputs/model_results/only_canon_5mirs/$${model_type}.txt \
			python train.py \
			--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
			--feature_file outputs/only_canon/features.txt \
			--out_folder outputs/model_results/only_canon_5mirs/ \
			--model_type $${model_type} ; \
	done


only_canon_sigmoid_models:
	for model_type in TS7_multisite_5mirs_logKA_noredundant_6canon_sigmoid \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid ; do \
		bsub -q 18 -o outputs/model_results/only_canon_5mirs/$${model_type}.txt \
			python train.py \
			--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
			--feature_file outputs/only_canon/features.txt \
			--out_folder outputs/model_results/only_canon_5mirs/ \
			--model_type $${model_type} ; \
	done


all_site_models:
	for model_type in logKA_sigmoid_freeago_allsites \
					  logKA_UTRlen_doublesigmoid_freeago_allsites \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_allsites \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_allsites_noPCT \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_freeago_allsites \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_freeago_allsites_3PNB ; do \
		bsub -q 18 -o outputs/model_results/all_sites_5mirs/$${model_type}.txt \
			python train.py \
			--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
			--feature_file outputs/all_sites_5mirs/features.txt \
			--out_folder outputs/model_results/all_sites_5mirs/ \
			--model_type $${model_type} ; \
	done


model_custom:
	mkdir -p ../outputs/biochem_model/logKA_UTRlen_doublesigmoid_freeago_allsites
	bsub -q 18 -o ../outputs/biochem_model/logKA_UTRlen_doublesigmoid_freeago_allsites/log.txt \
		python train.py \
		--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
		--feature_file ../data/biochem_model/all_sites_5mirs/features.txt \
		--out_folder ../outputs/biochem_model/logKA_UTRlen_doublesigmoid_freeago_allsites/ \
		--model_type logKA_UTRlen_doublesigmoid_freeago_allsites

model_custom_test:
	python train.py \
		--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
		--feature_file ../data/biochem_model/all_sites_6mirs/features.txt \
		--model_type TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_freeago_allsites_noORF8m



