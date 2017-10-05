# NN:
# 	cd targetscanNN && python targetscanNN.py -f ../../data/logfc_7mer_8mer_3p.txt -l ../../logdirs/TS -a ../../accuracy/TS_scores.txt -e 18

# pairing_seed:
# 	cd targetscanNN && python targetscanNN.py ../../data/logfc_3pairing_seed.txt 30565 3328 ../../logdirs/pairing ../../accuracy/pairing_scores.txt

# test:
# 	cd targetscanNN && python targetscanNN.py -f ../../data/count_match.txt -l ../../logdirs/testing -a ../../accuracy/testint_scores.txt

# flanking:
# 	cd targetscanNN && python flanking.py ../../data/logfc_flanking.txt 30454 ../../logdirs/flanking

# sample:
# 	cd targetscanNN && python sample.py -m 23 -s 12 -d 16000000 -a ../../accuracy/sample_scores.txt -l ../../logdirs/sample

# filter:
# 	cd targetscanNN && python filter.py -f ../../data/sean_data/ -m 20 -s 12 -o ../../data/sean_data/filtered.txt

kds:
	python regression/kd_regression/kd_regression.py -i ../data/sean_data/12mers/kds_multinomial.txt -l ../logdirs/2_layer_8_32_double_dropout2/ -n 10


kds2:
	for pos in 2-5 3-6 4-7 5-8 all; do \
		bsub python regression/kd_regression/kd_regression2.py -i ../data/sean_data/12mers/kds_multinomial_$${pos}.txt -l ../logdirs/kd_regression4/$${pos} -n 10 ; \
	done

# hyperparam:
# 	for pos in 2-5 3-6 4-7 5-8; do \
# 		bsub python regression/kd_regression/kd_regression_hyperparam.py -i ../data/sean_data/12mers/kds_multinomial_$${pos}.txt -n 1 ; \
# 	done

all:
	for pos in 2-5 3-6 4-7 5-8; do \
		bsub python regression/kd_regression/kd_regression_hyperparam.py -i ../data/sean_data/12mers/kds_multinomial_$${pos}.txt -l ../logdirs/train_all_4_8_$${pos}/ ; \
	done


test_positions:
	for pos in 2-5 3-6 4-7 5-8 ; do \
		bsub python regression/kd_regression/kd_regression.py -i ../data/sean_data/12mers/kds_multinomial_$${pos}.txt -l ../logdirs/test_positions5/$${pos} -n 10 ; \
	done

rbns:
	python regression/rbns_regression.py -i ../data/sean_data/12mers/match2-4.txt -l ../logdirs/rbns_match2-4/ -n 10

rbns_pretrained:
	python regression/rbns_regression.py -i ../data/sean_data/12mers/match2-4_any_register.txt -l ../logdirs/rbns_match2-4_any_register_pretrained/ -n 10

simple:
	python regression/rbns_regression.py -i ../data/simple/num_matches.txt -l ../logdirs/simple/num_matches/ -n 10

# repression:
# 	cd targetscanNN && python repression.py -i ../../data/transfections/one_site.txt -t 260 -m 20 -s 15 -l ../../logdirs/repression

# multiple:
# 	cd targetscanNN && \
# 	for layer3 in 1 2 3 4 5 6 7 8 9 10 ; do \
# 		bsub python rbns_multiple.py -i ../../data/sean_data/12mers/filtered_3match_shuffled.txt -t 5000 -m 20 -s 12 -l ../../logdirs/3_match_rbns/$${layer3} -y FALSE --l1 2 --l2 4 --l3 32 ; \
# 	done

# shuffle:
# 	cd targetscanNN \
# 	&& sort -R ../../data/sean_data/12mers/match2-4.txt > ../../data/sean_data/12mers/match2-4_shuffled.txt

# targeted:
# 	cd rbns_NN
# 	&& python rbns_multiple.py -i ../../data/sean_data/12mers/all_12mers_shuffled.txt -t 1000 -m 20 -s 12 -l ../../logdirs/match3 -y TRUE --l1 1 --l2 4 --l3 32 --old
# 	# cd targetscanNN && python rbns_multiple.py -i ../../data/sean_data/12mers/filtered_seed_match_shuffled.txt -t 500 -m 20 -s 12 -l ../../logdirs/3_match_rbns/9_moreruns -y FALSE --l1 2 --l2 4 --l3 32


# kd_regression:
# 	cd rbns_NN && \
# 	for run in 1 2 3 4 5 6 7 8 9 10 ; do \
# 		bsub python kd_regression.py -i ../../data/sean_data/12mers/kds_multinomial_shuffled.txt -t 1000 -m 20 -s 12 -l ../../logdirs/kds/kd_regression_4_8_32_$${run} -y TRUE --l1 4 --l2 8 --l3 32 ; \
# 	done

# train_kds:
# 	python rbns_NN/kd_regression.py -i ../data/sean_data/12mers/kds_multinomial_shuffled2.txt -t 10000 -m 20 -s 12 -l ../logdirs/kds/kd_regression_gradual -y TRUE --l1 4 --l2 16 --l3 16

# conv3:
# 	cd targetscanNN \
# 	&& python rbns_conv3.py -i ../../data/sean_data/12mers/all_12mers_shuffled.txt -t 1000 -m 20 -s 12 -l ../../logdirs/conv3 -y FALSE --l1 2 --l2 4 --l3 3


# kds:
# 	cd targetscanNN && python rbns_kds.py -i ../../data/sean_data/12mers/2-6kds_shuffled.txt -t 1000 -m 20 -s 12 -l ../../logdirs/kds -y False --l1 2 --l2 4 --l3 32


