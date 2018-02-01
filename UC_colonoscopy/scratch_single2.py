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

batch_size=32

train_datagen = ImageDataGenerator( 
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        path+'zero_nzero/train/',  # this is the target directory
        target_size=(256, 320),  
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        path+'zero_nzero/validation/',
        target_size=(256, 320),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 320, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())  
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

adam = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=model_path+'splitp_znz0201.h5', verbose=0, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=10, min_lr=1.e-7)
earlystop = EarlyStopping(monitor='loss', patience=30)

history = model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // (batch_size),
            epochs=200,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // (batch_size),
            callbacks = [checkpointer, reduce_lr, earlystop],
            verbose=2);

model.save_weights(model_path+'splitp_znz0201.h5')

with open('output/splitp_znz0201.pkl', 'wb') as f:
    pickle.dump(history.history, f, -1)
