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


class RepressionData(Data):

    def __init__(self, dataframe):
        super().__init__(dataframe)

    def shuffle(self):
        shuffle_ix = np.random.permutation(self.length)
        self.data = self.data.iloc[shuffle_ix]

class RepressionDataNew(Data):

    def __init__(self, dataframe):
        super().__init__(dataframe)

    def shuffle(self):
        shuffle_ix = np.random.permutation(self.length)
        self.data = self.data.iloc[shuffle_ix]

    def get_seqs(self, mirs):
        self.seq_dict = {}
        self.num_sites_dict = {}
        for row in self.data.iterrows():
            gene_dict = {}
            num_sites_gene = 0
            for mir in mirs:
                utr = row[1]['Sequence']

                seqs = helpers.get_seqs(utr, config.SITE_DICT[mir], only_canon=False)
                # gene_dict[mir] = [helpers.one_hot_encode_nt(seq, np.array(['T','A','G','C'])) for seq in seqs]
                gene_dict[mir] = seqs

                if len(seqs) > num_sites_gene:
                    num_sites_gene = len(seqs)

            self.num_sites_dict[row[0]] = num_sites_gene    
            self.seq_dict[row[0]] = gene_dict

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
        all_seqs = [[self.seq_dict[gene][mir] for mir in mirs] for gene in genes]
        num_sites = max([self.num_sites_dict[gene] for gene in genes])
        batch_y = next_batch[mirs].values

        return new_epoch, all_seqs, num_sites, batch_y


class RepressionDataNew_with_passenger(Data):

    def __init__(self, dataframe):
        super().__init__(dataframe)

    def shuffle(self):
        shuffle_ix = np.random.permutation(self.length)
        self.data = self.data.iloc[shuffle_ix]

    def get_seqs(self, mirs):
        self.seq_dict = {}
        self.num_sites_dict = {}
        self.num_sites_dict_pass = {}
        for row in self.data.iterrows():
            gene_dict = {}
            num_sites_gene = 0
            num_sites_gene_pass = 0
            for mir in mirs:
                utr = row[1]['Sequence']

                seqs = helpers.get_seqs(utr, config.SITE_DICT[mir], only_canon=False)
                seqs_pass = helpers.get_seqs(utr, config.SITE_DICT[mir + '*'], only_canon=False)
                gene_dict[mir] = seqs
                gene_dict[mir + '*'] = seqs_pass

                if len(seqs) > num_sites_gene:
                    num_sites_gene = len(seqs)

                if len(seqs_pass) > num_sites_gene_pass:
                    num_sites_gene_pass = len(seqs_pass)

            self.num_sites_dict[row[0]] = num_sites_gene 
            self.num_sites_dict_pass[row[0]] = num_sites_gene_pass  
            self.seq_dict[row[0]] = gene_dict

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
        all_seqs_site = [[self.seq_dict[gene][mir] for mir in mirs] for gene in genes]
        all_seqs_pass = [[self.seq_dict[gene][mir + '*'] for mir in mirs] for gene in genes]
        num_sites = max([self.num_sites_dict[gene] for gene in genes] + [self.num_sites_dict_pass[gene] for gene in genes])
        batch_y = next_batch[mirs].values

        return genes, new_epoch, all_seqs_site, all_seqs_pass, num_sites, batch_y


class BiochemData(Data):
    def __init__(self, dataframe, cutoff=0.9):
        super().__init__(dataframe)
        self.original_data = dataframe.copy()
        self.original_length = len(self.original_data)
        self.cutoff = cutoff

    def shuffle(self):
        assert (len(self.original_data) == self.original_length)
        self.original_data['keep'] = [(np.random.random() > self.cutoff) if x == 'grey' else True for x in self.original_data['color']]
        self.data = self.original_data[self.original_data['keep']]
        self.length = len(self.data)
        shuffle_ix = np.random.permutation(self.length)
        self.data = self.data.iloc[shuffle_ix]

