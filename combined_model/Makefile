features_test:
	python write_features.py \
		--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged_mini.txt \
		--kd_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/kds.txt \
		--orf_file /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_input_files/refseq_utr90_fixed/orf90.txt \
		--mirseqs mirseqs.txt \
		--pct /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_output_files/refseq_utr90_fixed/features.txt \
		--ta_sps TA_SPS_all.txt \
		--rnaplfold_folder /lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/RNAplfold_utr90_fixed_u15 \
		--outdir outputs/only_canon_test \
		--overlap_dist 7 \
		--upstream_limit 15 \
		--only_canon

features:
	python write_features.py \
		--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
		--kd_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/kds.txt \
		--orf_file /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_input_files/refseq_utr90_fixed/orf90.txt \
		--mirseqs mirseqs.txt \
		--pct /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_output_files/refseq_utr90_fixed/features.txt \
		--ta_sps TA_SPS_all.txt \
		--rnaplfold_folder /lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/RNAplfold_utr90_fixed_u15 \
		--outdir outputs/all_sites \
		--overlap_dist 7 \
		--upstream_limit 15

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
					  TS7_multisite \
					  TS7_multisite_5mirs \
					  TS7_multisite_5mirs_logKA ; do \
		bsub -q 18 -o outputs/all_sites/$${model_type}.txt \
			python train.py \
			--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
			--feature_file outputs/all_sites/features.txt \
			--out_folder outputs/all_sites/ \
			--model_type $${model_type} ; \
	done

linear_models:
	for model_type in TS7_multisite_5mirs_logKA_noredundant \
					  TS7_multisite_5mirs_logKA_noredundant_6canon ; do \
		bsub -q 18 -o outputs/all_sites/$${model_type}.txt \
			python train.py \
			--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
			--feature_file outputs/all_sites/features.txt \
			--out_folder outputs/all_sites/ \
			--model_type $${model_type} ; \
	done


sigmoid_models:
	for model_type in TS7_multisite_5mirs_logKA_noredundant_6canon_sigmoid \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_energy ; do \
		bsub -q 18 -o outputs/all_sites/$${model_type}.txt \
			python train.py \
			--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
			--feature_file outputs/all_sites/features.txt \
			--out_folder outputs/all_sites/ \
			--model_type $${model_type} ; \
	done


all_site_models:
	for model_type in logKA_sigmoid_freeago_allsites \
					  logKA_UTRlen_doublesigmoid_freeago_allsites \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_allsites \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_allsites_noPCT \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_freeago_allsites \
					  TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_freeago_allsites_3PNB ; do \
		bsub -q 18 -o outputs/all_sites/$${model_type}.txt \
			python train.py \
			--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
			--feature_file outputs/all_sites/features.txt \
			--out_folder outputs/all_sites/ \
			--model_type $${model_type} ; \
	done

model_custom:
	bsub -q 18 -o outputs/all_sites/TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_freeago_allsites_noORF8m.txt \
		python train.py \
			--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
			--feature_file outputs/all_sites/features.txt \
			--out_folder outputs/all_sites/ \
			--model_type TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_freeago_allsites_noORF8m

model_custom_test:
	python train.py \
		--tpm_file /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
		--feature_file outputs/all_sites/features.txt \
		--model_type TS7_multisite_5mirs_logKA_noredundant_6canon_doublesigmoid_allsites



