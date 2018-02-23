import numpy as np
import pandas as pd
import tensorflow as tf


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

