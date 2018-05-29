# data_dir="/lab/bartel4_ata/kathyl/NeuralNet/data/tpm_and_kds"
final_data_dir="/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/final"
# final_data_dir="/lab/bartel4_ata/kathyl/RNA_Seq/analysis/data/pre_final"

pretrain:
	python main.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m mir223 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_16_last

pretrain_pass:
	python main_with_pass_no_baseline.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt -m mir223 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_16_last_pass_mirlen_20 -p pretrain

train:
	python main_with_pass.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m mir124 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/testing_all_mirs -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_16_last_pass/pretrain_saved

train_no_pretrain:
	python main_with_pass.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m mir137 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/testing_mir137_no_pretrain -d

train_pass_no_baseline:
	python main_with_pass_no_baseline.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt --hidden1 4 --hidden2 8 --hidden3 16 -m none -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run_pass_fit_baseline_no_pretrain_testing/all_mirs_no_pretrain -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_16_last_pass/pretrain_saved

test:
	python main_labmeeting.py -k $(final_data_dir)/kds.txt -t $(final_data_dir)/merged.txt -i $(final_data_dir)/baselines.txt --hidden1 4 --hidden2 8 --hidden3 16 -m none -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run_pass_fit_baseline_second_run/testing -d

lab_meeting_version:
	python main_labmeeting.py -k $(final_data_dir)/kds.txt -t $(final_data_dir)/merged.txt -i $(final_data_dir)/baselines.txt --hidden1 4 --hidden2 8 --hidden3 16 -m none -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/testing -d


endog_pretrain:
	python main_endog.py -k $(final_data_dir)/kds.txt -t $(final_data_dir)/merged.txt --let7_sites $(final_data_dir)/let7_sites.txt --let7_mask $(final_data_dir)/let7_mask.txt -p $(final_data_dir)/prefit.txt --hidden1 4 --hidden2 16 --hidden3 16 -m mir1 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_pretrain_just_weights --pretrain pretrain


endog_use_pretrain:
	python main_endog.py -k $(final_data_dir)/kds.txt -t $(final_data_dir)/merged.txt --let7_sites $(final_data_dir)/let7_sites.txt --let7_mask $(final_data_dir)/let7_mask.txt -p $(final_data_dir)/prefit.txt --hidden1 4 --hidden2 16 --hidden3 32 -m mir1 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_mir1_pretrain_passrange1 --pretrain /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_pretrain_just_weights/pretrain_saved

endog_no_pretrain:
	python main_endog.py -k $(final_data_dir)/kds.txt -t $(final_data_dir)/merged.txt --let7_sites $(final_data_dir)/let7_sites.txt --let7_mask $(final_data_dir)/let7_mask.txt -p $(final_data_dir)/prefit.txt --hidden1 4 --hidden2 16 --hidden3 16 -m mir1 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_mir1_none --pretrain none

endog_reload:
	python main_endog.py -k $(final_data_dir)/kds.txt -t $(final_data_dir)/merged.txt --let7_sites $(final_data_dir)/let7_sites.txt --let7_mask $(final_data_dir)/let7_mask.txt -p $(final_data_dir)/prefit.txt --hidden1 4 --hidden2 16 --hidden3 16 -m mir1 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_mir1_tune --pretrain none -r /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_mir1_pretrain/saved

optimize:
	python optimize.py -k $(final_data_dir)/kds.txt -t $(final_data_dir)/merged.txt --let7_sites $(final_data_dir)/let7_sites.txt --let7_mask $(final_data_dir)/let7_mask.txt -p $(final_data_dir)/prefit.txt --hidden1 4 --hidden2 16 --hidden3 16 -m mir1 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_mir1_pretrain --pretrain none

#/lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_pretrain/pretrain_saved

endog_cross_val:
	mkdir /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_xval_trainable
	cp config.py /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_xval_trainable/config.txt
	for mirname in none mir1 mir124 mir155 mir7 lsy6 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		bsub -o /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_xval_trainable/$${mirname}/output.txt python main_endog.py -k $(final_data_dir)/kds.txt -t $(final_data_dir)/merged.txt --let7_sites $(final_data_dir)/let7_sites.txt --let7_mask $(final_data_dir)/let7_mask.txt -p $(final_data_dir)/prefit.txt --hidden1 4 --hidden2 16 --hidden3 32 -m $${mirname} -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_xval_trainable/$${mirname} --pretrain /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/endog_pretrain_just_weights/pretrain_saved; \
	done


lab_meeting_version_cross_val:
	mkdir /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run_pass_fit_baseline_fit_decay_final_epoch50_lr1-3
	cp config.py /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run_pass_fit_baseline_fit_decay_final_epoch50_lr1-3/config.txt
	for mirname in none mir1 mir124 mir155 mir7 lsy6 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		bsub -o /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run_pass_fit_baseline_fit_decay_final_epoch50_lr1-3/$${mirname}/output.txt python main_labmeeting.py -k $(final_data_dir)/kds.txt -t $(final_data_dir)/merged.txt -i $(final_data_dir)/baselines.txt --hidden1 4 --hidden2 8 --hidden3 16 -m $${mirname} -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run_pass_fit_baseline_fit_decay_final_epoch50_lr1-3/$${mirname} -d; \
	done


single_sample:
	python main.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt -m mir139 --hidden1 4 --hidden2 16 --hidden3 32 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run_pass_fit_baseline_no_pretrain_fit_slope_4_16_16_32_epoch50/mir139 -d


cross_val:
	for mirname in mir1 mir124 mir155 lsy6 mir7 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		bsub python main.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines_cutoff_0.txt -m $${mirname} -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run2/$${mirname} -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_8_last/pretrain_saved ; \
	done

cross_val_pass:
	for mirname in mir1 mir124 mir155 mir7 lsy6 mir153 mir139 mir204 mir216b mir223 mir137 mir199a mir205 mir182 mir143 mir144 ; do \
		bsub python main.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt -m $${mirname} --hidden1 4 --hidden2 16 --hidden3 32 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run_pass_no_pretrain_no_offset_fit_decay_4_16_16_32_epoch50/$${mirname} -d; \
	done

cross_val_rbns_pass:
	for mirname in mir1 mir124 mir155 mir7 lsy6 ; do \
		bsub python main.py -k $(data_dir)/kds.txt -t $(data_dir)/tpms.txt -i $(data_dir)/baselines.txt -m $${mirname} --hidden1 4 --hidden2 8 --hidden3 16 -l /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/run_pass_fit_baseline_pretrain_fit_decay_rbns_log1p_4_8_16_epoch50/$${mirname} -d -p /lab/bartel4_ata/kathyl/NeuralNet/logdirs/tpms_and_kds/pretrain_0_centered_normed_16_last_pass/pretrain_saved; \
	done