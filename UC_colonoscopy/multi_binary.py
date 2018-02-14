import os
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pickle
import numpy as np

path = "/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/UC_colonoscopy/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

batch_size=32

X_trn = np.load(path+'array_data/image_array_256_320_train.npy')
X_tst = np.load(path+'array_data/image_array_256_320_test.npy')
binary_label_trn = np.load(path+'array_data/binary_labels_train.npy')
binary_label_tst = np.load(path+'array_data/binary_labels_test.npy')

train_datagen = ImageDataGenerator( 
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(X_trn, binary_label_trn, batch_size=batch_size)
validation_generator = test_datagen.flow(X_tst, binary_label_tst, batch_size=batch_size)

from keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(3, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights(model_path+'binary_inceptionv3_trntop_f.h5')

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=model_path+'binary_inceptionv3_e2e0212.h5', verbose=0, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=1.e-8)
earlystop = EarlyStopping(monitor='loss', patience=10)

#class_weights = {0:(5895/4467.), 1:(5895./828), 2:(5895./389), 3:(5895./211)}

history = model.fit_generator(train_generator, steps_per_epoch=train_generator.n // batch_size, epochs=200, 
                              validation_data=validation_generator, validation_steps=validation_generator.n // batch_size, 
                              callbacks=[checkpointer, reduce_lr, earlystop], 
                                verbose=2)

with open('output/binary_inceptionv3_e2e0212.pkl', 'wb') as f:
    pickle.dump(history.history, f, -1)
    
model.save_weights(model_path+'binary_inceptionv3_e2e0212_f.h5')
