import numpy as np
import os
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Lambda, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pickle

path = "/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/UC_colonoscopy/"
model_path = path + 'models/'

batch_size=16
G = 4 # number of GPU's

train_datagen = ImageDataGenerator( 
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        path+'splitted/train/',  # this is the target directory
        target_size=(341, 427),  
        batch_size=batch_size*G,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        path+'splitted/validation/',
        target_size=(341, 427),
        batch_size=batch_size*G,
        shuffle=False,
        class_mode='categorical')

with tf.device('/cpu:0'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(341, 427, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())  
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    
parallel_model = multi_gpu_model(model, gpus=G)

adam = Adam()
parallel_model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=model_path+'scratch_hd0120.h5', verbose=0, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=1.e-7)
earlystop = EarlyStopping(patience=30)

history = parallel_model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // (batch_size*G),
            epochs=200,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // (batch_size*G),
            callbacks = [checkpointer, reduce_lr, earlystop],
            verbose=2);

with open('output/history_hd0120.pkl', 'wb') as f:
    pickle.dump(history.history, f, -1)
