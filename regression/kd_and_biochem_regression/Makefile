data_dir="/lab/bartel4_ata/kathyl/NeuralNet/data/tpm_and_kds"

pretrain:
	python main.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m mir223 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_16_last

pretrain_pass:
	python main_with_pass.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m mir223 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_16_last_pass

train:
	python main_with_pass.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m mir124 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/testing_all_mirs -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_16_last_pass/pretrain_saved

train_no_pretrain:
	python main_with_pass.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m mir137 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/testing_mir137_no_pretrain -d

train_pass_no_baseline:
	python main_with_pass_no_baseline.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt -m mir204 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run5_pass_no_baseline_4mers/all_mirs -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_16_last_pass/pretrain_saved

optimize:
	python optimize.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m mir139


cross_val:
	for mirname in mir1 mir124 mir155 lsy6 mir7 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		bsub python main.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m $${mirname} -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run2/$${mirname} -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_8_last/pretrain_saved ; \
	done


cross_val_pass:
	for mirname in mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		python main_with_pass_no_baseline.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt -m $${mirname} -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run5_pass_no_baseline_4mers/$${mirname} -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_16_last_pass/pretrain_saved ; \
	done