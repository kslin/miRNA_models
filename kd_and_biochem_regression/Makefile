dry_run:
	python main_with_feats.py \
		--tpm_tfrecords /lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/guide_passenger_only_canon.tfrecord \
		--kd_tfrecords /lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/log_kds \
		--mirseqs write_tfrecords/mirseqs.txt \
		--mirlen 10 \
		--num_feats 6 \
		--repression_batch_size 10 \
		--kd_batch_size 10 \
		--num_epochs 2 \
		--val_mir mir124 \
		--baseline bayes_baseline \
		--loss_type MEAN_CENTER \
		--lambda 0.001 \
		--lr 0.002 \
		--logdir /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/train_with_feats_val_mir124 \
		--passenger \
		--dry_run

simple:
	mkdir -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_only_canon
	bsub -o /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_mir124_with_neg_examples_4_16_16/output.txt \
	python main.py \
		--kdfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/kds.txt \
		--tpmfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged_with_baselines.txt \
		--mirna mir124 \
		--baseline baseline \
		--loss_type MEAN_CENTER \
		--logdir /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_mir124_with_neg_examples_4_16_16

simple_xval:
	mkdir /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_clean_only_canon
	cp config.py /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_clean_only_canon/config.txt
	for mirname in none mir1 mir124 mir155 mir7 lsy6 let7 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		bsub -o /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_clean_only_canon/output_$${mirname}.txt \
			python main.py \
			--kdfile  /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/kds.txt \
			--tpmfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged_with_baselines.txt \
			--mirna $${mirname} \
			--baseline bayes_baseline \
			--loss_type MEAN_CENTER \
			--logdir /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_clean_only_canon/$${mirname}; \
	done


simple_xval_rerun:
	for mirname in mir205 mir199a; do \
		bsub -o /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_only_canon/output_$${mirname}.txt \
			python main.py \
			--kdfile  /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/kds.txt \
			--tpmfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged_with_baselines.txt \
			--mirna $${mirname} \
			--baseline baseline \
			--loss_type MEAN_CENTER \
			--logdir /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_only_canon/$${mirname}; \
	done
