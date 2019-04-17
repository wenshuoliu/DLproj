import numpy as np
import pandas as pd
from keras.utils import Sequence

class DoubleBatchGenerator(Sequence):
    'Generates data from two datasets'
    def __init__(self, cooccur_df, readm_df, readm_batchsize=512, batchsize_ratio=16, shuffle=True, scaling_factor=0.75, 
                count_cap=20):
        '''Initialization
        Here coocur_df and readm_df need to have integer index from 0, i.e. they need to reset_index()'''
        self.readm_batchsize = readm_batchsize
        self.cooccur_batchsize = readm_batchsize*batchsize_ratio
        self.batchsize_ratio = batchsize_ratio
        self.cooccur_df = cooccur_df
        self.readm_df = readm_df
        self.shuffle = shuffle
        self.__scaling_factor = scaling_factor
        self.__count_cap = count_cap
        self.batch_size = self.readm_batchsize
        self.n = self.__len__()*self.batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        readm_steps = np.floor(len(self.readm_df) / self.readm_batchsize)
        cooccur_steps = np.floor(len(self.cooccur_df)/self.cooccur_batchsize)
        return int(np.min((readm_steps, cooccur_steps)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        readm_indexes = self.readm_indexes[index*self.readm_batchsize:(index+1)*self.readm_batchsize]
        cooccur_indexes = self.cooccur_indexes[index*self.cooccur_batchsize:(index+1)*self.cooccur_batchsize]

        DX_mat = self.readm_df.loc[readm_indexes, DXs].values
        PR_mat = self.readm_df.loc[readm_indexes, PRs].values
        continue_mat = self.readm_df.loc[readm_indexes, ['AGE', 'GENDER', 'LOS']].values
        eth_array = self.readm_df.loc[readm_indexes, 'Eth'].values
        eth_mat = to_categorical(eth_array, num_classes=5)
        other_mat = np.concatenate((continue_mat, eth_mat), axis=1)
        y_readm = self.readm_df.loc[readm_indexes, 'MORTALITY_1year'].astype(int).values
        Y_readm = to_categorical(y_readm, num_classes=2)

        focal_id = self.cooccur_df.loc[cooccur_indexes, 'focal_index'].values
        context_id = self.cooccur_df.loc[cooccur_indexes, 'context_index'].values
        y_cooccur = np.log(self.cooccur_df.loc[cooccur_indexes, 'cooccur_counts'].values)
        
        focal_id = np.reshape(focal_id, (self.readm_batchsize, self.batchsize_ratio))
        context_id = np.reshape(context_id, (self.readm_batchsize, self.batchsize_ratio))
        y_cooccur = np.reshape(y_cooccur, (self.readm_batchsize, self.batchsize_ratio))

        return [DX_mat, PR_mat, other_mat, focal_id, context_id], [Y_readm, y_cooccur]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.readm_indexes = np.arange(len(self.readm_df))
        self.cooccur_indexes = np.arange(len(self.cooccur_df))
        if self.shuffle == True:
            np.random.shuffle(self.readm_indexes)
            np.random.shuffle(self.cooccur_indexes)