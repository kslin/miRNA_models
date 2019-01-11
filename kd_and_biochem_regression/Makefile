simple:
	python main_simple.py \
		--kdfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/kds.txt \
		--tpmfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged_with_baselines.txt
		--mirna mir124 \
		--baseline baseline \
		--loss_type MEAN_CENTER \
		--logdir /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_mir124_with_neg_examples_4_16_16

# simple_xval:
# 	mkdir /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_neg_examples
# 	cp config.py /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_neg_examples/config.txt
# 	for mirname in none mir1 mir124 mir155 mir7 lsy6 let7 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
# 		bsub -o /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_neg_examples/output_$${mirname}.txt python main_simple.py -k /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/kds.txt -t /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged_with_baselines.txt -m $${mirname} --pretrain none -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/simple_xval_4_16_16_neg_examples/$${mirname}; \
# 	done
