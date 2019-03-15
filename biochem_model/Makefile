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



