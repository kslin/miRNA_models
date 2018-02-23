data_dir="/lab/bartel4_ata/kathyl/NeuralNet/data/tpm_and_kds"

pretrain:
	python main.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m mir223 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_5_centered_normed_16_last

train:
	python main.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m mir1 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/testing_mir1 -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_5_centered_normed_16_last_0_cutoff/pretrain_saved


cross_val:
	for mirname in mir7 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		bsub python main.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m $${mirname} -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/$${mirname} -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_zero_centered_normed/pretrain_saved ; \
	done