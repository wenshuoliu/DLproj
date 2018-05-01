""" Tune the hyper-parameters. """
import pandas as pd
import numpy as np
import os, sys
from sklearn.metrics import roc_curve, auc

from keras.layers import Input, Embedding, concatenate, Reshape, BatchNormalization, add
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam

path = '/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/NRD/'
model_path = path + 'models/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)
    
data = np.load(path+'temp/data_temp.npz')
demo_mat = data['demo_mat']
DX1_mat = data['DX1_mat'] 
DX_mat = data['DX_mat']
hosp_array = data['hosp_array']
y_train = data['y_train'] 
y_test = data['y_test']
n_DX_cat = data['n_DX_cat']
n_DX1_cat = data['n_DX1_cat']
n_hosp_cat = data['n_hosp_cat']
N_train = y_train.shape[0]

#print(demo_mat.shape, DX1_mat.shape, DX_mat.shape, hosp_array.shape, y_train.shape, y_test.shape, n_DX_cat, n_DX1_cat, 
#      n_hosp_cat, N_train)

depths = [1, 3, 5, 10, 20, 50]
widths = [64, 128, 256, 512, 1024]
DX_dims = [5, 10, 20, 50]
hosp_dims = [1, 3, 5, 10]

n_sample = 100
parameters = list(zip(np.random.choice(depths, n_sample), np.random.choice(widths, n_sample),
                       np.random.choice(DX_dims, n_sample), np.random.choice(hosp_dims, n_sample)))
parameters = [p for p in parameters if parameters.count(p)==1]
param_tried = pd.read_csv('./hyper_tune.csv')
param_tried = param_tried[['depth', 'width', 'DX_dim', 'hosp_dim']].values
param_tried = [tuple(p) for p in list(param_tried)]

def sc_block(input_tensor, width):
    '''Short-cut block'''
    x = Dense(width)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Dense(width)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Dense(width)(x)
    x = BatchNormalization()(x)
    
    x = add([input_tensor, x])
    x = Activation('relu')(x)
    return x

def model_build(depth, width, DX_dim, hosp_dim):
    input_DX = Input(shape = (29,))
    DX_embed = Embedding(input_dim=n_DX_cat, output_dim=DX_dim, input_length=29)(input_DX)
    DX_embed = Reshape((29*DX_dim,))(DX_embed)

    input_demo = Input(shape=(2, ))
    input_DX1 = Input(shape=(n_DX1_cat,))

    input_hosp = Input(shape=(1,))
    hosp_embed = Embedding(input_dim=n_hosp_cat, output_dim=hosp_dim, input_length=1)(input_hosp)
    hosp_embed = Reshape((hosp_dim, ))(hosp_embed)

    merged = concatenate([input_demo, input_DX1, DX_embed, hosp_embed], axis=1)
    x = Dense(width, activation='relu')(merged)

    for j in range(depth):
        x = sc_block(x, width)

    x = Dense(width, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(width, activation='relu')(x)
    prediction = Dense(2, activation='softmax')(x)
    model = Model(inputs=[input_demo, input_DX1, input_DX, input_hosp], outputs=prediction)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class_weight = {0:(y_train.shape[0]/sum(y_train[:, 0])), 1:(y_train.shape[0]/sum(y_train[:, 1]))}
adam = Adam(lr=0.0002)

#with open('./hyper_tune.csv', 'a') as f:
#    f.write('depth,width,DX_dim,hosp_dim,auc\n')
for param in parameters:
    if not param in param_tried: 
        print("Training with parameters: ", *param)
        model = model_build(*param)
        checkpointer = ModelCheckpoint(filepath=model_path+'ami_icd9_{}_{}_{}_{}.h5'.format(*param), verbose=0, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1.e-8)
        earlystop = EarlyStopping(monitor='val_loss', patience=20)
        hist = model.fit([demo_mat[:N_train, :], DX1_mat[:N_train, :], DX_mat[:N_train, :], hosp_array[:N_train]], y_train, 
                         batch_size=256, epochs=100, callbacks=[checkpointer, reduce_lr, earlystop], class_weight=class_weight, 
                         validation_data=[[demo_mat[N_train:, :], DX1_mat[N_train:, :], DX_mat[N_train:, :], hosp_array[N_train:]], y_test], 
                         verbose=2)
        model.load_weights(model_path+'ami_icd9_{}_{}_{}_{}.h5'.format(*param))
        y_pred = model.predict([demo_mat[N_train:, :], DX1_mat[N_train:, :], DX_mat[N_train:, :], hosp_array[N_train:]], batch_size=256)
        fpr, tpr, _ = roc_curve(y_test[:, 0], y_pred[:, 0])
        roc_auc = auc(fpr, tpr)
        with open('./hyper_tune.csv', 'a') as f:
            f.write('{0},{1},{2},{3},{4:.3f}\n'.format(*param,roc_auc))
        del(model)

