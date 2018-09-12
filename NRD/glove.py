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
from utils import Parent_reg

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
        
    def get_cooccur_dict(self):
        if len(self.__cooccur_counts) == 0:
            raise ValueError('Co-occurrences is empty!')
        return self.__cooccur_counts
    
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
        
    def train_glove(self, cooccur_df=None, cache_path='./', batch_size=512, epochs=50, earlystop_patience=20, reducelr_patience=10, validation_split=0.2, verbose=1, parent_pairs=None, lamb=1e-3, norm=2):
        print('Preparing data...')
        if cooccur_df is None:
            cooccur_df = self.get_cooccur_df()
        focal_id = cooccur_df.focal_index.values
        context_id = cooccur_df.context_index.values
        y = np.log(cooccur_df.cooccur_counts.values)
        weights = cooccur_df.cooccur_counts.apply(self.__weighting_factor).values
        
        print('Defining the GloVe model...')
        input_w = Input(shape=(1,), name='focal_index')
        input_v = Input(shape=(1,), name='context_index')
        if parent_pairs is None:
            embed_layer = Embedding(input_dim=self.__input_dim, output_dim=self.__embed_dim, name='embed')
        else:
            embed_layer = Embedding(input_dim=self.__input_dim, output_dim=self.__embed_dim, name='embed',
                                embeddings_regularizer=Parent_reg(parent_pairs, lamb, norm))
        w_embed = embed_layer(input_w)
        v_embed = embed_layer(input_v)
        bias_layer = Embedding(input_dim=self.__input_dim, output_dim=1, name='bias')
        w_bias = bias_layer(input_w)
        v_bias = bias_layer(input_v)
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
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=reducelr_patience, min_lr=K.epsilon())
        earlystop = EarlyStopping(monitor='val_loss', patience=earlystop_patience)
        
        print('Training the GloVe model...')
        hist = model.fit([focal_id, context_id], y, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                         sample_weight=weights, callbacks=[checkpoint, reduce_lr, earlystop], verbose=verbose)
        self.train_history = hist
        
        model.load_weights(cache_path+'glove_temp.h5')
        for l in model.layers:
            if l.name=='embed':
                embed_mat = l.get_weights()[0]
        os.remove(cache_path+'glove_temp.h5')
        self.__embed_mat = embed_mat
        print('Finished. The pretrained embedding matrix can be retrieved by .get_embed_mat().')
        
    def train_glove_double(self, cooccur_df=None, cache_path='./', batch_size=512, epochs=50, earlystop_patience=20, reducelr_patience=10, validation_split=0.2, verbose=1, focal_initializer='uniform', context_initializer='uniform', focal_trainable=True, context_trainable=True):
        '''This function train two sets of embedding: one for focal word, one for context word. It has the option to freeze one of them. 
        '''
        print('Preparing data...')
        if cooccur_df is None:
            cooccur_df = self.get_cooccur_df()
        focal_id = cooccur_df.focal_index.values
        context_id = cooccur_df.context_index.values
        y = np.log(cooccur_df.cooccur_counts.values)
        weights = cooccur_df.cooccur_counts.apply(self.__weighting_factor).values
        
        print('Defining the GloVe model...')
        input_w = Input(shape=(1,), name='focal_index')
        input_v = Input(shape=(1,), name='context_index')
        w_embed = Embedding(input_dim=self.__input_dim, output_dim=self.__embed_dim, name='focal_embed', 
                            embeddings_initializer=focal_initializer)(input_w)
        v_embed = Embedding(input_dim=self.__input_dim, output_dim=self.__embed_dim, name='context_embed', 
                            embeddings_initializer=context_initializer)(input_v)
        w_bias = Embedding(input_dim=self.__input_dim, output_dim=1, name='focal_bias')(input_w)
        v_bias = Embedding(input_dim=self.__input_dim, output_dim=1, name='context_bias')(input_v)
        w_embed = Reshape((self.__embed_dim, ))(w_embed)
        v_embed = Reshape((self.__embed_dim, ))(v_embed)
        w_bias = Reshape((1,))(w_bias)
        v_bias = Reshape((1,))(v_bias)
        inner = Multiply()([w_embed, v_embed])
        inner = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(inner)
        merged = Concatenate(axis=1)([inner, w_bias, v_bias])
        out = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(merged)
        model = Model(inputs=[input_w, input_v], outputs = out)
        
        for l in model.layers:
            if l.name=='focal_embed':
                l.trainable = focal_trainable
            if l.name=='context_embed':
                l.trainable = context_trainable
        
        model.compile(optimizer='adam', loss='mse')
        
        checkpoint = ModelCheckpoint(filepath=cache_path+'glove_temp.h5', save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=reducelr_patience, min_lr=K.epsilon())
        earlystop = EarlyStopping(monitor='val_loss', patience=earlystop_patience)
        
        print('Training the GloVe model...')
        hist = model.fit([focal_id, context_id], y, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                         sample_weight=weights, callbacks=[checkpoint, reduce_lr, earlystop], verbose=verbose)
        self.train_history = hist
        
        model.load_weights(cache_path+'glove_temp.h5')
        for l in model.layers:
            if l.name=='focal_embed':
                focal_embed_mat = l.get_weights()[0]
            if l.name=='context_embed':
                context_embed_mat = l.get_weights()[0]
        os.remove(cache_path+'glove_temp.h5')
        self.__focal_embed_mat = focal_embed_mat
        self.__context_embed_mat = context_embed_mat
        print('Finished. The pretrained embedding matrix can be retrieved by .get_embed_mat().')
        
    def get_embed_mat(self, double=False):
        if double:
            return self.__focal_embed_mat, self.__context_embed_mat
        else:
            return self.__embed_mat
                