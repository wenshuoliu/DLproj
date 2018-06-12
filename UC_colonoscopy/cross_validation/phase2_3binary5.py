import os, sys
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from keras_addon import ImageFrameGenerator, AUCCheckPoint

path  =  "/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/colonoscopy2/"
model_path = path + 'models/cross_validation/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)
data_path = path+'subset/'

batch_size=32

labels = pd.read_csv(path+'cross_validation/train_labels5.csv')
labels = labels.reset_index(drop=True)
split = GroupShuffleSplit(n_splits=1, test_size=0.11, random_state=24)
ind = split.split(labels, groups=labels['SourceReportName'])
trn_ind, val_ind = next(ind)
trn_df = labels.loc[trn_ind, ]
val_df = labels.loc[val_ind, ]

train_gen = ImageFrameGenerator( 
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
test_gen = ImageFrameGenerator()
trn_itr = train_gen.flow_from_frame(data_path, trn_df, 'basefile', ['split0_123', 'split01_23', 'split012_3'], 
                                     target_size=(256, 320), label_types = ['categorical']*3, batch_size=batch_size)
val_itr = test_gen.flow_from_frame(data_path, val_df, 'basefile', ['split0_123', 'split01_23', 'split012_3'], 
                                     target_size=(256, 320), label_types = ['categorical']*3, batch_size=batch_size, shuffle=False)

from keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)

output1 = Dense(2, activation='softmax', name='split0_123')(x)
output2 = Dense(2, activation='softmax', name='split01_23')(x)
output3 = Dense(2, activation='softmax', name='split012_3')(x)

model = Model(inputs=base_model.input, outputs=[output1, output2, output3])

adam = Adam()
model.compile(optimizer=adam, loss='categorical_crossentropy')

checkpointer = ModelCheckpoint(filepath=model_path+'binary3_valloss5.h5', verbose=0, save_best_only=True, 
                               save_weights_only=True)
auccheckpt = AUCCheckPoint(filepath=model_path+'binary3_auc5.h5', 
                           validation_y=val_df[['split0_123', 'split01_23', 'split012_3']].values,
                          validation_itr=val_itr)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1.e-8)
earlystop = EarlyStopping(monitor='val_loss', patience=30)

class_weight = {'split0_123':{0:len(trn_df)/sum(trn_df.split0_123==0), 1:len(trn_df)/sum(trn_df.split0_123==1)}, 
               'split01_23':{0:len(trn_df)/sum(trn_df.split01_23==0), 1:len(trn_df)/sum(trn_df.split01_23==1)}, 
               'split012_3':{0:len(trn_df)/sum(trn_df.split012_3==0), 1:len(trn_df)/sum(trn_df.split012_3==1)}}

hist = model.fit_generator(trn_itr, steps_per_epoch=trn_itr.n // batch_size, epochs=200, 
                              validation_data=val_itr, validation_steps=val_itr.n // batch_size, 
                              callbacks=[checkpointer, auccheckpt, reduce_lr, earlystop], 
                                verbose=2)

#with open('output/binary3_0528.pkl', 'wb') as f:
#    pickle.dump(hist.history, f, -1)
    
#with open('output/binary3_auc_0528.pkl', 'wb') as f:
#    pickle.dump(auccheckpt.auc_history, f, -1)
