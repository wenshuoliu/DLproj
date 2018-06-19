#model training
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling3D, Input, ZeroPadding3D, BatchNormalization, MaxPooling3D, Concatenate, AveragePooling3D
from keras.layers.core import Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Conv3D
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import multi_gpu_model
import tensorflow as tf

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from keras_addon import ImageFrameGenerator, AUCCheckPoint

#path = '/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/CT_scans/'
path = '/nfs/turbo/umms-awaljee/wsliu/Data/CT_scans/'
model_path = path + 'models/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)
    
batch_size = 4
G = 4
tot_bs = batch_size*G

df = pd.read_csv(path+'train_labels.csv')

val_size = int(len(df)*0.125/tot_bs)*tot_bs
trn_size = int((len(df)-val_size)/tot_bs)*tot_bs

trn_df, val_df = train_test_split(df, test_size=val_size, train_size=trn_size, stratify=df[['CD_Active_AnyLocation', 'Fistula_Any', 'Abscess_any']], random_state=24)

gen = ImageFrameGenerator()
trn_itr = gen.flow_from_frame(path+'ndarray/', trn_df, 'filename', ['CD_Active_AnyLocation', 'Fistula_Any', 'Abscess_any'], 
                             target_size=(224, 224, 128), color_mode='3d', batch_size=batch_size*G, shuffle=True)
val_itr = gen.flow_from_frame(path+'ndarray/', val_df, 'filename', ['CD_Active_AnyLocation', 'Fistula_Any', 'Abscess_any'], 
                             target_size=(224, 224, 128), color_mode='3d', batch_size=batch_size*G, shuffle=False)

from dense3dnet import Dense3DNet
blocks = [6, 12, 24, 16]

with tf.device('/cpu:0'):
    base_model = Dense3DNet(blocks, growth_rate=12, input_shape=(224, 224, 128, 1), pooling='avg')
    x = base_model.output
    output_CD = Dense(1, activation='sigmoid', name='CD_Active_AnyLocation')(x)
    output_fist = Dense(1, activation='sigmoid', name='Fistula_Any')(x)
    output_absc = Dense(1, activation='sigmoid', name='Abscess_any')(x)
    model = Model(inputs=base_model.input, outputs=[output_CD, output_fist, output_absc])
parallel_model = multi_gpu_model(model, gpus=G)

parallel_model.compile(optimizer='adam', loss={'CD_Active_AnyLocation':'binary_crossentropy', 'Fistula_Any':'binary_crossentropy', 
                                     'Abscess_any':'binary_crossentropy'}, metrics=['accuracy'], 
             loss_weights={'CD_Active_AnyLocation':1., 'Fistula_Any':1., 'Abscess_any':1.})

#parallel_model.compile(optimizer='adam', loss={'Abscess_any':'binary_crossentropy'}, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=model_path+'dense121_gr12_3output0611.h5', verbose=0, save_best_only=True, save_weights_only=True)
auccheckpoint = AUCCheckPoint(filepath=model_path+'dense121_gr12_3output_auc0611.h5', validation_y=val_df[['CD_Active_AnyLocation', 'Fistula_Any', 'Abscess_any']].values, validation_itr=val_itr)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1.e-8)
earlystop = EarlyStopping(monitor='val_loss', patience=30)

hist = parallel_model.fit_generator(trn_itr, steps_per_epoch=trn_itr.n // (batch_size*G), epochs=200, validation_data=val_itr, 
                    validation_steps=val_itr.n // (batch_size*G), callbacks=[checkpointer, auccheckpoint, reduce_lr, earlystop], verbose=2)

#parallel_model.save_weights(model_path+'dense121_gr12_3output0515_f.h5')

with open('output/dense121_gr12_3output0611.pkl', 'wb') as f:
    pickle.dump(hist.history, f, -1)

with open('output/dense121_gr12_3output0611_auc.pkl', 'wb') as f:
    pickle.dump(auccheckpoint.auc_history, f, -1)
