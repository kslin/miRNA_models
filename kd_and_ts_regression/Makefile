tfrecords_utr:
	for mirname in none mir1 mir124 mir155 mir7 lsy6 let7 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		bsub python write_tfrecords_utr.py \
			--merged /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged.txt \
			--orffile /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_input_files/refseq_utr90/human_orf_90.txt \
			--test_mirna $${mirname} \
			--rnaplfold_folder /lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/RNAplfold_utr90_u14 \
			--outfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/$${mirname}_with_passenger.tfrecord \
			--overlap_dist 7 \
			--upstream_limit 15 \
			--passenger ; \
	done

tfrecords_mir124:
	python write_tfrecords_utr.py \
		--merged /lab/bartel4_ata/kathyl/RNA_Seq/outputs/biochem/final/merged.txt \
		--orffile /lab/bartel4_ata/kathyl/RNA_Seq/data/TS7_input_files/refseq_utr90/human_orf_90.txt \
		--test_mirna mir124 \
		--rnaplfold_folder /lab/bartel4_ata/kathyl/RNA_Seq/outputs/rnaplfold/RNAplfold_utr90_u14 \
		--outfile /lab/bartel4_ata/kathyl/RNA_Seq/outputs/convnet/tfrecords/mir124_with_passenger.tfrecord \
		--overlap_dist 7 \
		--upstream_limit 15 \
		--passenger