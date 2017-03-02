NN:
	cd targetscanNN && python targetscanNN.py -f ../../data/logfc_7mer_8mer_3p.txt -l ../../logdirs/TS -a ../../accuracy/TS_scores.txt -e 18

pairing_seed:
	cd targetscanNN && python targetscanNN.py ../../data/logfc_3pairing_seed.txt 30565 3328 ../../logdirs/pairing ../../accuracy/pairing_scores.txt

test:
	cd targetscanNN && python targetscanNN.py -f ../../data/count_match.txt -l ../../logdirs/testing -a ../../accuracy/testint_scores.txt

flanking:
	cd targetscanNN && python flanking.py ../../data/logfc_flanking.txt 30454 ../../logdirs/flanking

sample:
	cd targetscanNN && python sample.py -m 23 -s 12 -d 16000000 -a ../../accuracy/sample_scores.txt -l ../../logdirs/sample

rbns:
	cd targetscanNN && python rbns.py -f ../../data/sean_data/ -i ../../data/sean_counts_train.txt -t 100000 -m 20 -s 12 -l ../../logdirs/rbns