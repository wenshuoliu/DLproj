# Implement the GloVe by keras
from collections import defaultdict
import numpy as np
import pandas as pd
import os
from keras.layers import Input, Embedding, Concatenate, Reshape, Lambda, Multiply, Add
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.initializers import Constant
import keras.backend as K
import time
from utils import Parent_reg
from keras.engine.topology import InputSpec, Layer

class Glove(object):
    """ The class to train embedding by GloVe.
    The user need to provide a pandas Dataframe of codes in terms of integers. 
    """
    def __init__(self, input_dim=None, embedding_dim=None, scaling_factor=0.75, count_cap=100):
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
    
    def _weighting_factor(self, count):
        if count>=self.__count_cap:
            return 1.
        else:
            return np.power(count/self.__count_cap, self.__scaling_factor)
        
    def train_glove(self, cooccur_df=None, cache_path='./', batch_size=512, lr=1e-3, epochs=50, earlystop_patience=20, reducelr_patience=10, verbose=1, parent_pairs=None, lamb=1., metric='l2'):
        print('Preparing data...')
        if cooccur_df is None:
            cooccur_df = self.get_cooccur_df()
        focal_id = cooccur_df.focal_index.values
        context_id = cooccur_df.context_index.values
        y = np.log(cooccur_df.cooccur_counts.values)
        weights = cooccur_df.cooccur_counts.apply(self._weighting_factor).values
        
        print('Defining the GloVe model...')
        input_w = Input(shape=(1,), name='focal_index')
        input_v = Input(shape=(1,), name='context_index')
        if parent_pairs is None:
            embed_layer = Embedding(input_dim=self.__input_dim, output_dim=self.__embed_dim, name='embed')
        else:
            embed_layer = Embedding(input_dim=self.__input_dim, output_dim=self.__embed_dim, name='embed',
                                embeddings_regularizer=Parent_reg(parent_pairs, lamb, metric))
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
        
        adam = Adam(lr=lr)
        model.compile(optimizer=adam, loss='mse')
        
        checkpoint = ModelCheckpoint(filepath=cache_path+'glove_temp.h5', monitor='loss', save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=reducelr_patience, min_lr=K.epsilon())
        earlystop = EarlyStopping(monitor='loss', patience=earlystop_patience)
        
        print('Training the GloVe model...')
        hist = model.fit([focal_id, context_id], y, batch_size=batch_size, epochs=epochs,
                         sample_weight=weights, callbacks=[checkpoint, reduce_lr, earlystop], verbose=verbose)
        self.train_history = hist
        
        model.load_weights(cache_path+'glove_temp.h5')
        for l in model.layers:
            if l.name=='embed':
                embed_mat = l.get_weights()[0]
            if l.name=='bias':
                bias_mat = l.get_weights()[0]
        os.remove(cache_path+'glove_temp.h5')
        self.__embed_mat = embed_mat
        self.__bias_mat = bias_mat
        print('Finished. The pretrained embedding matrix can be retrieved by .get_embed_mat().')
        
    def train_glove_double(self, cooccur_df=None, cache_path='./', batch_size=512, epochs=50, earlystop_patience=20, reducelr_patience=10, verbose=1, focal_initializer='uniform', context_initializer='uniform', focal_trainable=True, context_trainable=True):
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
        
        checkpoint = ModelCheckpoint(filepath=cache_path+'glove_temp.h5', monitor='loss', save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=reducelr_patience, min_lr=K.epsilon())
        earlystop = EarlyStopping(monitor='loss', patience=earlystop_patience)
        
        print('Training the GloVe model...')
        hist = model.fit([focal_id, context_id], y, batch_size=batch_size, epochs=epochs,
                         sample_weight=weights, callbacks=[checkpoint, reduce_lr, earlystop], verbose=verbose)
        self.train_history = hist
        
        model.load_weights(cache_path+'glove_temp.h5')
        for l in model.layers:
            if l.name=='focal_embed':
                focal_embed_mat = l.get_weights()[0]
            if l.name=='context_embed':
                context_embed_mat = l.get_weights()[0]
            if l.name=='focal_bias':
                focal_bias_mat = l.get_weights()[0]
            if l.name=='context_bias':
                context_bias_mat = l.get_weights()[0]
        os.remove(cache_path+'glove_temp.h5')
        self.__focal_embed_mat = focal_embed_mat
        self.__context_embed_mat = context_embed_mat
        self.__focal_bias_mat = focal_bias_mat
        self.__context_bias_mat = context_bias_mat
        print('Finished. The pretrained embedding matrix can be retrieved by .get_embed_mat().')
        
    def get_embed_mat(self, double=False):
        if double:
            return self.__focal_embed_mat, self.__context_embed_mat
        else:
            return self.__embed_mat
                
    def get_bias_mat(self, double=False):
        if double:
            return self.__focal_bias_mat, self.__context_bias_mat
        else:
            return self.__bias_mat
        
class InnerProduct(Layer):
    """The layer to perform a inner-product of two vectors through a matrix.   

    # Arguments
        matrix_shape: (dim1, dim2), shape of the matrix between the two vectors. 
        kernel_initializer: Initializer for the `kernel` weights matrix
    # Input shape
        [(batch, dim1), (batch, dim2)]
    # Output shape
        (batch, 1). 
    """
    def __init__(self, matrix_shape=None,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(InnerProduct, self).__init__(**kwargs)
        self.matrix_shape = matrix_shape
        self.kernel_initializer = kernel_initializer
        self.supports_masking = True

    def build(self, input_shape):
        input_shape1, input_shape2 = input_shape
        if self.matrix_shape is None:
            self.matrix_shape = (input_shape1[-1], input_shape2[-1])
        elif not self.matrix_shape == (input_shape1[-1], input_shape2[-1]):
            raise ValueError('Initializer shape conflicts with inpute shape!')

        self.kernel = self.add_weight(shape=self.matrix_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel')
        self.built = True

    def call(self, inputs, mask=None):
        if not mask is None:
            mask1, mask2 = mask
        input1, input2 = inputs
        output = K.dot(input1, self.kernel)*input2
        output = K.sum(output, axis=-1)
        #mask1 = K.expand_dims(mask1)
        #mask2 = K.expand_dims(mask2)
        #print(output.shape, mask1.shape, mask2.shape)
        output = output*K.cast(mask1, 'float32')*K.cast(mask2, 'float32')
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        input_shape1, input_shape2 = input_shape
        assert input_shape1[0] == input_shape2[0]
        return (input_shape1[0], 1)
    
    def compute_mask(self, inputs, input_mask=None):
        return None

    def get_config(self):
        config = {
            'matrix_shape': self.matrix_shape,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Masking(Layer):
    """The layer to multipy the input with its mask.   
    # Arguments
    # Input shape
        Input can be a tensor with any shape. The last layer should pass a mask with (1) the same shape as the input; 
        (2) one less dimension than the input. In this case the input has to have shape (batch, ..., 1), and the mask
        has shape (batch, ...), which is the case when the last layer is Embedding(input_length=1, mask_zero=True). 
        
    # Output shape
        The same as input shape.  
    """
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Masking, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, mask=None):
        if mask is None:
            return inputs
        if K.ndim(mask)<K.ndim(inputs):
            mask_expand = K.expand_dims(mask)
        else:
            mask_expand = mask
        if not inputs.shape[1:]==mask_expand.shape[1:]:
            raise ValueError("The shapes of inputs and mask don't match!")
        output = inputs*K.cast(mask_expand, 'float32')
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def compute_mask(self, inputs, input_mask=None):
        return None
    
class GloveMS(Glove):
    """ The child class of Glove, to embed DX1, DX and PR into different spaces, and train them with GloVe. 
    The user need to provide a pandas Dataframe of codes in terms of integers. 
    """
    def __init__(self, n_DX1_cat, n_DX_cat, n_PR_cat, DX1_dim, DX_dim, PR_dim, n_dx1_ccs=None, n_dx_ccs=None, n_pr_ccs=None, **kwargs):
            super(GloveMS, self).__init__(**kwargs)
            self.__n_DX1_cat = n_DX1_cat
            self.__n_DX_cat = n_DX_cat
            self.__n_PR_cat = n_PR_cat
            self.__DX1_dim = DX1_dim
            self.__DX_dim = DX_dim
            self.__PR_dim = PR_dim
            if n_dx1_ccs is None:
                self.__regularize = False
            else:
                self.__regularize = True
                self.__n_dx1_ccs = n_dx1_ccs
                self.__n_dx_ccs = n_dx_ccs
                self.__n_pr_ccs = n_pr_ccs
            
    def train_glove(self, cooccur_df=None, cache_path='./', batch_size=512, lr=1e-3, epochs=50, optimizer='adam', earlystop_patience=20, reducelr_patience=10, dx1_parent_pairs=None, dx_parent_pairs=None, pr_parent_pairs=None, lamb=1., metric='l2', verbose=2):
        print('Preparing data...')
        if cooccur_df is None:
            cooccur_df = self.get_cooccur_df()
        input_X = cooccur_df.copy()
        input_X = input_X.assign(input_DX1_w=0).assign(input_DX_w=0).assign(input_PR_w=0).assign(input_DX1_v=0).assign(input_DX_v=0).assign(input_PR_v=0)
        input_X.loc[input_X.context_index<=self.__n_DX1_cat, 'input_DX1_w'] = input_X.loc[input_X.context_index<=self.__n_DX1_cat, 'context_index']
        input_X.loc[(self.__n_DX1_cat<input_X.context_index) & (input_X.context_index<=self.__n_DX1_cat+self.__n_DX_cat), 'input_DX_w'] = input_X.loc[(self.__n_DX1_cat<input_X.context_index) & (input_X.context_index<=self.__n_DX1_cat+self.__n_DX_cat), 'context_index'] - self.__n_DX1_cat
        input_X.loc[input_X.context_index>self.__n_DX1_cat+self.__n_DX_cat, 'input_PR_w'] = input_X.loc[input_X.context_index>self.__n_DX1_cat+self.__n_DX_cat, 'context_index'] - self.__n_DX1_cat - self.__n_DX_cat
        input_X.loc[input_X.focal_index<=self.__n_DX1_cat, 'input_DX1_v'] = input_X.loc[input_X.focal_index<=self.__n_DX1_cat, 'focal_index']
        input_X.loc[(self.__n_DX1_cat<input_X.focal_index) & (input_X.focal_index<=self.__n_DX1_cat+self.__n_DX_cat), 'input_DX_v'] = input_X.loc[(self.__n_DX1_cat<input_X.focal_index) & (input_X.focal_index<=self.__n_DX1_cat+self.__n_DX_cat), 'focal_index'] - self.__n_DX1_cat
        input_X.loc[input_X.focal_index>self.__n_DX1_cat+self.__n_DX_cat, 'input_PR_v'] = input_X.loc[input_X.focal_index>self.__n_DX1_cat+self.__n_DX_cat, 'focal_index'] - self.__n_DX1_cat - self.__n_DX_cat
        DX1_w_trn = input_X['input_DX1_w'].values
        DX1_v_trn = input_X['input_DX1_v'].values
        DX_w_trn = input_X['input_DX_w'].values
        DX_v_trn = input_X['input_DX_v'].values
        PR_w_trn = input_X['input_PR_w'].values
        PR_v_trn = input_X['input_PR_v'].values
        y = np.log(input_X.cooccur_counts.values)
        weights = input_X.cooccur_counts.apply(self._weighting_factor).values
        
        print('Defining the GloVe model...')
        input_DX1_w = Input(shape=(1,), name='input_focal_DX1')
        input_DX_w = Input(shape=(1,), name='input_focal_DX')
        input_PR_w = Input(shape=(1,), name='input_focal_PR')
        input_DX1_v = Input(shape=(1,), name='input_context_DX1')
        input_DX_v = Input(shape=(1,), name='input_context_DX')
        input_PR_v = Input(shape=(1,), name='input_context_PR')

        if self.__regularize==False:
            DX1_embed_layer = Embedding(input_dim=self.__n_DX1_cat+1, output_dim=self.__DX1_dim, name='DX1_embed', mask_zero=True)
            DX_embed_layer = Embedding(input_dim=self.__n_DX_cat+1, output_dim=self.__DX_dim, name='DX_embed', mask_zero=True)
            PR_embed_layer = Embedding(input_dim=self.__n_PR_cat+1, output_dim=self.__PR_dim, name='PR_embed', mask_zero=True)
        else:
            DX1_embed_layer = Embedding(input_dim=self.__n_DX1_cat+self.__n_dx1_ccs+1, output_dim=self.__DX1_dim, name='DX1_embed',
                                        mask_zero=True, embeddings_regularizer=Parent_reg(dx1_parent_pairs, lamb, metric))
            DX_embed_layer = Embedding(input_dim=self.__n_DX_cat+self.__n_dx_ccs+1, output_dim=self.__DX_dim, name='DX_embed', 
                                       mask_zero=True, embeddings_regularizer=Parent_reg(dx_parent_pairs, lamb, metric))
            PR_embed_layer = Embedding(input_dim=self.__n_PR_cat+self.__n_pr_ccs+1, output_dim=self.__PR_dim, name='PR_embed', 
                                       mask_zero=True, embeddings_regularizer=Parent_reg(pr_parent_pairs, lamb, metric))
            
        DX1_bias_layer = Embedding(input_dim=self.__n_DX1_cat+1, output_dim=1, name='DX1_bias', mask_zero=True)
        DX_bias_layer = Embedding(input_dim=self.__n_DX_cat+1, output_dim=1, name='DX_bias', mask_zero=True)
        PR_bias_layer = Embedding(input_dim=self.__n_PR_cat+1, output_dim=1, name='PR_bias', mask_zero=True)            

        embed_DX1_w = DX1_embed_layer(input_DX1_w)
        embed_DX1_v = DX1_embed_layer(input_DX1_v)
        embed_DX_w = DX_embed_layer(input_DX_w)
        embed_DX_v = DX_embed_layer(input_DX_v)
        embed_PR_w = PR_embed_layer(input_PR_w)
        embed_PR_v = PR_embed_layer(input_PR_v)

        bias_DX1_w = DX1_bias_layer(input_DX1_w)
        bias_DX1_v = DX1_bias_layer(input_DX1_v)
        bias_DX_w = DX_bias_layer(input_DX_w)
        bias_DX_v = DX_bias_layer(input_DX_v)
        bias_PR_w = PR_bias_layer(input_PR_w)
        bias_PR_v = PR_bias_layer(input_PR_v)

        bias_DX1_w = Masking()(bias_DX1_w)
        bias_DX1_v = Masking()(bias_DX1_v)
        bias_DX_w = Masking()(bias_DX_w)
        bias_DX_v = Masking()(bias_DX_v)
        bias_PR_w = Masking()(bias_PR_w)
        bias_PR_v = Masking()(bias_PR_v)

        inner_DX1_DX = InnerProduct(matrix_shape=(self.__DX1_dim, self.__DX_dim), name='DX1dotDX')([embed_DX1_w, embed_DX_v])
        inner_DX1_PR = InnerProduct(matrix_shape=(self.__DX1_dim, self.__PR_dim), name='DX1dotPR')([embed_DX1_w, embed_PR_v])
        inner_DX_DX = InnerProduct(matrix_shape=(self.__DX_dim, self.__DX_dim), kernel_initializer=Constant(np.eye(self.__DX_dim)), name='DXdotDX')([embed_DX_w, embed_DX_v])
        inner_PR_PR = InnerProduct(matrix_shape=(self.__PR_dim, self.__PR_dim), kernel_initializer=Constant(np.eye(self.__PR_dim)), name='PRdotPR')([embed_PR_w, embed_PR_v])
        inner_DX_PR = InnerProduct(matrix_shape=(self.__DX_dim, self.__PR_dim), name='DXdotPR')([embed_DX_w, embed_PR_v])

        bias_concat = Concatenate(axis=-1)([bias_DX1_w, bias_DX1_v, bias_DX_w, bias_DX_v, bias_PR_w, bias_PR_v])
        bias_sum = Lambda(lambda x:K.sum(x, axis=-1, keepdims=False))(bias_concat)
        inner_bias_sum = Add()([inner_DX1_DX, inner_DX1_PR, inner_DX_DX, inner_DX_PR, inner_PR_PR, bias_sum])

        model = Model(inputs=[input_DX1_w, input_DX_w, input_PR_w, input_DX1_v, input_DX_v, input_PR_v], outputs=inner_bias_sum)
        
        for l in model.layers:
            if (l.name=='DXdotDX') or (l.name=='PRdotPR'):
                l.trainable = False
                
        model.compile(loss='mse', optimizer='adam')
        checkpoint = ModelCheckpoint(filepath=cache_path+'glovems_temp.h5', monitor='loss', save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=reducelr_patience, min_lr=K.epsilon())
        earlystop = EarlyStopping(monitor='loss', patience=earlystop_patience)
        
        print('Training the GloVe model...')
        hist = model.fit([DX1_w_trn, DX_w_trn, PR_w_trn, DX1_v_trn, DX_v_trn, PR_v_trn], y, batch_size=batch_size, epochs=epochs,
                         sample_weight=weights, callbacks=[checkpoint, reduce_lr, earlystop], verbose=verbose)
        self.train_history = hist
        
        model.load_weights(cache_path+'glovems_temp.h5')
        for l in model.layers:
            if l.name=='DX1_embed':
                self.__DX1_embed_mat = l.get_weights()[0]
            if l.name=='DX_embed':
                self.__DX_embed_mat = l.get_weights()[0]
            if l.name=='PR_embed':
                self.__PR_embed_mat = l.get_weights()[0]
            if l.name=='DX1_bias':
                self.__DX1_bias_mat = l.get_weights()[0]
            if l.name=='DX_bias':
                self.__DX_bias_mat = l.get_weights()[0]
            if l.name=='PR_bias':
                self.__PR_bias_mat = l.get_weights()[0]
            if l.name=='DX1dotDX':
                self.__DX1dotDX_mat = l.get_weights()[0]
            if l.name=='DX1dotPR':
                self.__DX1dotPR_mat = l.get_weights()[0]
            if l.name=='DXdotPR':
                self.__DXdotPR_mat = l.get_weights()[0]
        os.remove(cache_path+'glovems_temp.h5')
        print('Finished. The pretrained embedding matrix can be retrieved by .get_embed_mat().')

    def get_embed_mat(self):
        """Function to output all trained embeddings and biases
        """
        return {'DX1_embed':self.__DX1_embed_mat, 'DX_embed':self.__DX_embed_mat, 'PR_embed':self.__PR_embed_mat, 'DX1_bias':self.__DX1_bias_mat, 'DX_bias':self.__DX_bias_mat, 'PR_bias':self.__PR_bias_mat}
    
    def get_matrices(self):
        return {'DX1dotDX':self.__DX1dotDX_mat, 'DX1dotPR':self.__DX1dotPR_mat, 'DXdotPR':self.__DXdotPR_mat}