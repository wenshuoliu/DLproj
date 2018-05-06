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
from sklearn.model_selection import train_test_split
import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from keras_addon import ImageFrameGenerator

#path = '/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/CT_scans/'
path = '/nfs/turbo/umms-awaljee/wsliu/Data/CT_scans/'
model_path = path + 'models/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)
    
batch_size = 2
G = 4

df = pd.read_excel(path+'K23_Crohn_RadiologyReport_Labels_27MAR2018.xlsx')
df = df.dropna(subset=['StudyID'])
df['filename'] = ['s'+str(int(f)) for f in df['StudyID']]
df = df.drop_duplicates(subset=['filename'])
trn_df, tst_df = train_test_split(df, test_size=0.2, stratify=df[['CD_Active_AnyLocation', 'Fistula_Any', 'Abscess_any']])

gen = ImageFrameGenerator()
trn_gen = gen.flow_from_frame(path+'ndarray/', trn_df, 'filename', ['CD_Active_AnyLocation', 'Fistula_Any', 'Abscess_any'], 
                             target_size=(256, 256, 192), color_mode='3d', batch_size=batch_size*G)
tst_gen = gen.flow_from_frame(path+'ndarray/', tst_df, 'filename', ['CD_Active_AnyLocation', 'Fistula_Any', 'Abscess_any'], 
                             target_size=(256, 256, 192), color_mode='3d', batch_size=batch_size*G)

from dense3dnet import Dense3DNet
blocks = [6, 12, 24, 16]
model = Dense3DNet(blocks, growth_rate=12)
with tf.device('/cpu:0'):
    base_model = Dense3DNet(blocks, pooling='avg')
    x = base_model.output
    output_CD = Dense(1, activation='sigmoid', name='CD_Active_AnyLocation')(x)
    output_fist = Dense(1, activation='sigmoid', name='Fistula_Any')(x)
    output_absc = Dense(1, activation='sigmoid', name='Abscess_any')(x)
    model = Model(inputs=base_model.input, outputs=[output_CD, output_fist, output_absc])
parallel_model = multi_gpu_model(model, gpus=G)
parallel_model.compile(optimizer='adam', loss={'CD_Active_AnyLocation':'binary_crossentropy', 'Fistula_Any':'binary_crossentropy', 
                                     'Abscess_any':'binary_crossentropy'}, metrics=['accuracy'], 
             loss_weights={'CD_Active_AnyLocation':0.4, 'Fistula_Any':0.3, 'Abscess_any':0.3})

checkpointer = ModelCheckpoint(filepath=model_path+'dense121_gr12.h5', verbose=0, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1.e-8)
earlystop = EarlyStopping(monitor='val_loss', patience=15)

hist = parallel_model.fit_generator(trn_gen, steps_per_epoch=trn_gen.n // (batch_size*G), epochs=100, validation_data=tst_gen, 
                    validation_steps=tst_gen.n // (batch_size*G), callbacks=[checkpointer, reduce_lr, earlystop], verbose=2)