from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.applications.inception_v3 import InceptionV3

import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import cohen_kappa_score
import pickle

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from keras_addon import ImageFrameGenerator, AUCCheckPoint

from DL_utils import plot_roc

path  =  "/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/colonoscopy2/"
model_path = path + 'models/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)
data_path = path+'subset_adjudication/'

batch_size = 32

train_df = pd.read_csv(path+'train_labels.csv')
tst_df = pd.read_csv(path+'tst_labels.csv')

train_gen = ImageFrameGenerator( 
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

validation_gen = ImageFrameGenerator()

trn_itr = train_gen.flow_from_frame(data_path, train_df, 'basefile', ['split0_123', 'split01_23', 'split012_3'], 
                                     target_size=(256, 320), label_types = ['categorical']*3, batch_size=batch_size)
val_itr = validation_gen.flow_from_frame(data_path, tst_df, 'basefile', ['split0_123', 'split01_23', 'split012_3'], 
                                     target_size=(256, 320), label_types = ['categorical']*3, batch_size=batch_size, 
                                         shuffle=False)

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
output1 = Dense(2, activation='softmax', name='split0_123')(x)
output2 = Dense(2, activation='softmax', name='split01_23')(x)
output3 = Dense(2, activation='softmax', name='split012_3')(x)
model = Model(inputs=base_model.input, outputs=[output1, output2, output3])

model.compile(optimizer='adam', loss='categorical_crossentropy')

checkpointer = ModelCheckpoint(filepath=model_path+'binary3_valloss_all.h5', verbose=0, save_best_only=True, 
                               save_weights_only=True)
auccheckpt = AUCCheckPoint(filepath=model_path+'binary3_auc_all.h5', validation_itr=val_itr)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=K.epsilon())
earlystop = EarlyStopping(monitor='val_loss', patience=30)

class_weight = {'split0_123':{0:len(train_df)/sum(train_df.split0_123==0), 1:len(train_df)/sum(train_df.split0_123==1)}, 
               'split01_23':{0:len(train_df)/sum(train_df.split01_23==0), 1:len(train_df)/sum(train_df.split01_23==1)}, 
               'split012_3':{0:len(train_df)/sum(train_df.split012_3==0), 1:len(train_df)/sum(train_df.split012_3==1)}}

history = model.fit_generator(trn_itr, steps_per_epoch=trn_itr.n // batch_size, epochs=100, 
                              validation_data=val_itr, validation_steps=val_itr.n // batch_size, 
                              callbacks=[checkpointer, auccheckpt, reduce_lr, earlystop], class_weight = class_weight, 
                                verbose=2)