dry_run:
	python main_with_feats.py \
		--tpm_tfrecords ../data/cnn/tfrecords/guide_passenger_only_canon.tfrecord \
		--kd_tfrecords ../data/cnn/tfrecords/log_kds_canon \
		--mirseqs ../data/mirseqs.txt \
		--mirlen 10 \
		--hidden1 4 \
		--hidden2 16 \
		--hidden3 16 \
		--num_feats 6 \
		--repression_batch_size 40 \
		--kd_batch_size 10 \
		--num_epochs 2 \
		--val_mir mir124 \
		--baseline bayes_baseline \
		--loss_type MEAN_CENTER \
		--lambda 0.001 \
		--lr 0.002 \
		--logdir ../outputs/train_with_feats_val_mir124 \
		--passenger \
		--dry_run

train:
	python main_with_feats.py \
		--tpm_tfrecords ../data/cnn/tfrecords/guide_passenger_only_canon.tfrecord \
		--kd_tfrecords ../data/cnn/tfrecords/log_kds_canon \
		--mirseqs ../data/mirseqs.txt \
		--mirlen 10 \
		--hidden1 4 \
		--hidden2 16 \
		--hidden3 16 \
		--num_feats 6 \
		--repression_batch_size 40 \
		--kd_batch_size 10 \
		--num_epochs 100 \
		--val_mir mir124 \
		--baseline bayes_baseline \
		--loss_type MEAN_CENTER \
		--lambda 0.001 \
		--lr 0.002 \
		--logdir ../outputs/train_with_feats_val_mir124 \
		--passenger

simple:
	mkdir -p ../outputs/simple_mir124_with_neg_examples_4_16_16
	python main.py \
		--kdfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/kds.txt \
		--tpmfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final_fixed/merged.txt \
		--mirseqs ../data/mirseqs.txt \
		--test_mirna mir124 \
		--logdir ../outputs/simple_mir124_with_neg_examples_4_16_16

simple_xval:
	mkdir ../outputs/simple_xval_4_16_16_clean_only_canon
	cp config.py ../outputs/simple_xval_4_16_16_clean_only_canon/config.txt
	for mirname in none mir1 mir124 mir155 mir7 lsy6 let7 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		bsub -o ../outputs/simple_xval_4_16_16_clean_only_canon/output_$${mirname}.txt \
			python main.py \
			--kdfile  /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/kds.txt \
			--tpmfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged_with_baselines.txt \
			--mirna $${mirname} \
			--baseline bayes_baseline \
			--loss_type MEAN_CENTER \
			--logdir ../outputs/simple_xval_4_16_16_clean_only_canon/$${mirname}; \
	done


simple_xval_rerun:
	for mirname in mir205 mir199a; do \
		bsub -o ../outputs/simple_xval_4_16_16_only_canon/output_$${mirname}.txt \
			python main.py \
			--kdfile  /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/kds.txt \
			--tpmfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged_with_baselines.txt \
			--mirna $${mirname} \
			--baseline baseline \
			--loss_type MEAN_CENTER \
			--logdir ../outputs/simple_xval_4_16_16_only_canon/$${mirname}; \
	done
