tf_records_utr_mini:
	python write_tfrecords_utr.py \
		--tpmfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged_with_baselines_mini.txt \
		--mirseqs mirseqs.txt \
		--rnaplfold_folder /lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/RNAplfold_utr90_u14 \
		--outfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/only_tpms_mini.tfrecord \
		--mirlen 3 \
		--overlap_dist 7 \
		--upstream_limit 15 \
		--passenger \
		--only_canon \
		--write_seqs real_seqs.txt ; \

tf_records_utr_full:
	python write_tfrecords_utr.py \
		--tpmfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged_with_baselines.txt \
		--mirseqs mirseqs.txt \
		--rnaplfold_folder /lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/RNAplfold_utr90_u14 \
		--outfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/guide_passenger_only_canon.tfrecord \
		--mirlen 10 \
		--overlap_dist 7 \
		--upstream_limit 15 \
		--passenger \
		--only_canon ; \

tf_records_kds:
	python write_tfrecords_kds.py \
		--kdfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/kds.txt \
		--outfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/log_kds.tfrecord \
		--mirlen 10 \
		--only_canon ; \
