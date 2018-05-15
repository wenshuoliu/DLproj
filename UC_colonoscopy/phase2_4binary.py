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

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from keras_addon import ImageFrameGenerator

path  =  "/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/colonoscopy2/"
model_path = path + 'models/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)
data_path = path+'subset/'

batch_size=32

labels = pd.read_csv(path+'train_labels.csv')

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
train_itr = train_gen.flow_from_frame(data_path, trn_df, 'basefile', 
                                      ['Inflamm_Mayo_0', 'Inflamm_Mayo_1', 'Inflamm_Mayo_2', 'Inflamm_Mayo_3'], 
                                     target_size=(256, 320), batch_size=batch_size, shuffle=True)
val_itr = test_gen.flow_from_frame(data_path, val_df, 'basefile', 
                                   ['Inflamm_Mayo_0', 'Inflamm_Mayo_1', 'Inflamm_Mayo_2', 'Inflamm_Mayo_3'], 
                                     target_size=(256, 320), batch_size=batch_size, shuffle=False)

from keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)

output0 = Dense(1, activation='sigmoid', name='Inflamm_Mayo_0')(x)
output1 = Dense(1, activation='sigmoid', name='Inflamm_Mayo_1')(x)
output2 = Dense(1, activation='sigmoid', name='Inflamm_Mayo_2')(x)
output3 = Dense(1, activation='sigmoid', name='Inflamm_Mayo_3')(x)

model = Model(inputs=base_model.input, outputs=[output0, output1, output2, output3])

adam = Adam()
model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=model_path+'phase2_4binary0515.h5', verbose=0, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=1.e-8)
earlystop = EarlyStopping(monitor='val_loss', patience=30)

#class_weights = {0:(5895/4467.), 1:(5895./828), 2:(5895./389), 3:(5895./211)}

hist = model.fit_generator(train_itr, steps_per_epoch=train_itr.n // batch_size, epochs=200, 
                              validation_data=val_itr, validation_steps=val_itr.n // batch_size, 
                              callbacks=[checkpointer, reduce_lr, earlystop], 
                                verbose=2)

model.save_weights(model_path+'phase2_4binary0515_f.h5')

with open('output/phase2_4binary0515.pkl', 'wb') as f:
    pickle.dump(hist.history, f, -1)
