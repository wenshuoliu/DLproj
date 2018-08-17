# Implement the GloVe by keras
from collections import defaultdict
import numpy as np
import pandas as pd
import os
from keras.layers import Input, Embedding, Concatenate, Reshape, Lambda, Multiply
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import keras.backend as K
import time

class Glove(object):
    """ The class to train embedding by GloVe.
    The user need to provide a pandas Dataframe of codes in terms of integers. 
    """
    def __init__(self, input_dim, embedding_dim, scaling_factor=0.75, count_cap=100):
            self.__input_dim = input_dim
            self.__embed_dim = embedding_dim
            self.__scaling_factor = scaling_factor
            self.__count_cap = count_cap
            self.__cooccur_counts = defaultdict(float)
    
    def update_cooccur(self, int_df, mask_zero=True):
        """ This function calcualte the cooccurrences from the int_df, and add them to the cooccurrence dict of the class. 
        """
        start = time.time()
        print('Updating co-occurrence matrix from a Dataframe with {} rows...'.format(len(int_df)))
        for _, row in int_df.iterrows():
            code_s = set(row.values)
            if mask_zero:
                code_s.discard(0)
            code_l = sorted(list(code_s))
            for i in range(len(code_l)):
                for j in range(i):
                    self.__cooccur_counts[(code_l[i], code_l[j])] += 1
        print('Finished. It takes {:.1f} seconds to update the cooccurrences.'.format(time.time()-start))
        
    def get_cooccur_df(self):
        if len(self.__cooccur_counts) == 0:
            raise ValueError('Co-occurrences is empty!')
        pairs = list(self.__cooccur_counts.keys())
        counts = list(self.__cooccur_counts.values())
        focal, context = zip(*pairs)
        cooccur_df = pd.DataFrame(dict(focal_index=focal, context_index=context, cooccur_counts=counts), 
                          columns=['focal_index', 'context_index', 'cooccur_counts'])
        return cooccur_df
    
    def __weighting_factor(self, count):
        if count>=self.__count_cap:
            return 1.
        else:
            return np.power(count/self.__count_cap, self.__scaling_factor)
        
    def train_glove(self, cooccur_df=None, cache_path='./', batch_size=512, epochs=50, earlystop_patience=15, validation_split=0.2, verbose=1):
        print('Preparing data...')
        if cooccur_df is None:
            cooccur_df = self.get_cooccur_df()
        focal_id = np.concatenate([cooccur_df.focal_index.values, cooccur_df.context_index.values])
        context_id = np.concatenate([cooccur_df.context_index.values, cooccur_df.focal_index.values])
        y = np.log(np.tile(cooccur_df.cooccur_counts.values, 2))
        weights = np.tile(cooccur_df.cooccur_counts.apply(self.__weighting_factor).values, 2)
        
        print('Defining the GloVe model...')
        input_w = Input(shape=(1,), name='focal_index')
        input_v = Input(shape=(1,), name='context_index')
        w_embed = Embedding(input_dim=self.__input_dim, output_dim=self.__embed_dim, name='w_embed')(input_w)
        v_embed = Embedding(input_dim=self.__input_dim, output_dim=self.__embed_dim, name='v_embed')(input_v)
        w_bias = Embedding(input_dim=self.__input_dim, output_dim=1, name='w_bias')(input_w)
        v_bias = Embedding(input_dim=self.__input_dim, output_dim=1, name='v_bias')(input_v)
        w_embed = Reshape((self.__embed_dim, ))(w_embed)
        v_embed = Reshape((self.__embed_dim, ))(v_embed)
        w_bias = Reshape((1,))(w_bias)
        v_bias = Reshape((1,))(v_bias)
        inner = Multiply()([w_embed, v_embed])
        inner = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(inner)
        merged = Concatenate(axis=1)([inner, w_bias, v_bias])
        out = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(merged)
        model = Model(inputs=[input_w, input_v], outputs = out)
        
        model.compile(optimizer='adam', loss='mse')
        
        checkpoint = ModelCheckpoint(filepath=cache_path+'glove_temp.h5', save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=K.epsilon())
        earlystop = EarlyStopping(monitor='val_loss', patience=earlystop_patience)
        
        print('Training the GloVe model...')
        hist = model.fit([focal_id, context_id], y, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                         sample_weight=weights, callbacks=[checkpoint, reduce_lr, earlystop], verbose=verbose)
        self.train_history = hist
        
        model.load_weights(cache_path+'glove_temp.h5')
        for l in model.layers:
            if l.name=='w_embed':
                w_embed_mat = l.get_weights()[0]
            elif l.name=='v_embed':
                v_embed_mat = l.get_weights()[0]
        embed_mat = w_embed_mat + v_embed_mat
        os.remove(cache_path+'glove_temp.h5')
        self.__embed_mat = embed_mat
        print('Finished.')
        
    def get_embed_mat(self):
        return self.__embed_mat
                