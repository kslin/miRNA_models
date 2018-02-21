data_dir="/lab/bartel4_ata/kathyl/NeuralNet/data/tpm_and_kds"
train:
	python main3.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt -m mir182 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds


pretrain:
	python main4.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt -m mir223 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain

train1:
	python main4.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt -m mir139 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/mir139 -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain/pretrain_saved


cross_val:
	for mirname in mir7 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		bsub python main4.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt -m $${mirname} -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/$${mirname} -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain/pretrain_saved ; \
	done