tf_records_utr_mini:
	python write_tfrecords_utr.py \
		--tpmfile /Users/kathy/Lab/NeuralNet/data/merged_with_baselines_mini.txt \
		--mirseqs mirseqs.txt \
		--rnaplfold_folder /Users/kathy/Lab/NeuralNet/data/RNAplfold_utr90_u14 \
		--outfile /Users/kathy/Lab/NeuralNet/tfrecords/only_tpms_mini.tfrecord \
		--mirlen 3 \
		--overlap_dist 7 \
		--upstream_limit 15 \
		--passenger \
		--only_canon \
		--write_seqs real_seqs.txt ; \

tf_records_utr_full:
	python write_tfrecords_utr.py \
		--tpmfile /Users/kathy/Lab/NeuralNet/data/merged_with_baselines.txt \
		--mirseqs mirseqs.txt \
		--rnaplfold_folder /Users/kathy/Lab/NeuralNet/data/RNAplfold_utr90_u14 \
		--outfile /Users/kathy/Lab/NeuralNet/tfrecords/guide_passenger_only_canon.tfrecord \
		--mirlen 10 \
		--overlap_dist 7 \
		--upstream_limit 15 \
		--passenger \
		--only_canon ; \

tf_records_kds:
	python write_tfrecords_kds.py \
		--kdfile /Users/kathy/Lab/NeuralNet/data/kds.txt \
		--outfile /Users/kathy/Lab/NeuralNet/tfrecords/log_kds_only_canon \
		--mirlen 10 \
		--only_canon ; \
