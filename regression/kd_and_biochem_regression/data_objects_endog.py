import numpy as np
import pandas as pd
import tensorflow as tf

import config, helpers


class Data:
    def __init__(self, dataframe):
        self.data = dataframe.copy()
        self.current_ix = 0
        self.length = len(dataframe)
        self.num_epochs = 0

    def shuffle(self):
        raise NotImplementedError()

    def get_next_batch(self, batch_size):
        new_epoch = False
        if (self.length - self.current_ix) < batch_size:
            self.shuffle()
            self.current_ix = 0
            self.num_epochs += 1
            new_epoch = True

        next_batch = self.data.iloc[self.current_ix: self.current_ix + batch_size]
        self.current_ix += batch_size

        return new_epoch, next_batch


class BiochemData(Data):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.original_data = dataframe.copy()
        # self.original_data['keep_prob'] = 1 / (1 + np.exp(-1 * (self.original_data['log ka'] - 2)))
        self.original_length = len(self.original_data)

    def shuffle(self):
        assert (len(self.original_data) == self.original_length)
        self.data = self.original_data[[np.random.random() < x for x in self.original_data['keep_prob']]]
        self.length = len(self.data)
        shuffle_ix = np.random.permutation(self.length)
        self.data = self.data.iloc[shuffle_ix]


class RepressionData(Data):

    def __init__(self, dataframe):
        super().__init__(dataframe)

    def shuffle(self):
        shuffle_ix = np.random.permutation(self.length)
        self.data = self.data.iloc[shuffle_ix]

    def get_seqs(self, mirs, only_canon):
        self.seq_dict = {}
        self.num_sites_dict = {}
        self.num_sites_dict_pass = {}
        for row in self.data.iterrows():
            gene_dict = {}
            num_sites_gene = 0
            num_sites_gene_pass = 0
            for mir in mirs:
                utr = row[1]['Sequence']

                seqs = helpers.get_seqs_new(utr, config.SITE_DICT[mir], only_canon=only_canon)
                seqs_pass = helpers.get_seqs_new(utr, config.SITE_DICT[mir + '*'], only_canon=only_canon)
                gene_dict[mir] = seqs
                gene_dict[mir + '*'] = seqs_pass

                if len(seqs) > num_sites_gene:
                    num_sites_gene = len(seqs)

                if len(seqs_pass) > num_sites_gene_pass:
                    num_sites_gene_pass = len(seqs_pass)

            self.num_sites_dict[row[0]] = num_sites_gene 
            self.num_sites_dict_pass[row[0]] = num_sites_gene_pass  
            self.seq_dict[row[0]] = gene_dict

    def get_next_batch_no_shuffle(self, batch_size, mirs):
        new_epoch = False
        if (self.length - self.current_ix) < batch_size:
            next_batch = self.data.iloc[self.current_ix:]
            self.current_ix = 0
            self.num_epochs += 1
            new_epoch = True

        else:
            next_batch = self.data.iloc[self.current_ix: self.current_ix + batch_size]
            self.current_ix += batch_size

        genes = list(next_batch.index)
        all_seqs, num_sites = [], []
        for gene in genes:
            for mir in mirs:
                seqs_guide, seqs_pass = self.seq_dict[gene][mir], self.seq_dict[gene][mir + '*']
                all_seqs.append((seqs_guide, seqs_pass))
                num_sites += [len(seqs_guide), len(seqs_pass)]
        
        max_sites = np.max(num_sites)
        batch_y = next_batch[mirs].values

        return genes, new_epoch, all_seqs, np.array(num_sites), max_sites, batch_y

    def get_next_batch(self, batch_size, mirs):
        new_epoch = False
        if (self.length - self.current_ix) < batch_size:
            self.shuffle()
            self.current_ix = 0
            self.num_epochs += 1
            new_epoch = True

        next_batch = self.data.iloc[self.current_ix: self.current_ix + batch_size]
        self.current_ix += batch_size

        genes = list(next_batch.index)
        all_seqs, num_sites = [], []
        for gene in genes:
            for mir in mirs:
                seqs_guide, seqs_pass = self.seq_dict[gene][mir], self.seq_dict[gene][mir + '*']
                all_seqs.append((seqs_guide, seqs_pass))
                num_sites += [len(seqs_guide), len(seqs_pass)]
        
        max_sites = np.max(num_sites)
        batch_y = next_batch[mirs].values

        return genes, new_epoch, all_seqs, np.array(num_sites), max_sites, batch_y