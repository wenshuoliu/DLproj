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

path = "/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/UC_colonoscopy/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

batch_size=64

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

train_generator = train_datagen.flow_from_directory(
        path+'split_patients/train/',  # this is the target directory
        target_size=(256, 320),  
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        path+'split_patients/validation/',
        target_size=(256, 320),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')


from keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights(model_path+'inceptionv3_trn_top_f.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=model_path+'inceptionv3_ft.h5', verbose=0, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=20, min_lr=1.e-8)
earlystop = EarlyStopping(patience=50)

class_weights = {0:(5895/4467.), 1:(5895./828), 2:(5895./389), 3:(5895./211)}

history = model.fit_generator(train_gen, steps_per_epoch=train_gen.n // batch_size, epochs=200, validation_data=test_gen, 
                    validation_steps=test_gen.n // batch_size, callbacks=[checkpointer, reduce_lr, earlystop], 
                             class_weight=class_weights)

with open('output/inceptionv3_ft.pkl', 'wb') as f:
    pickle.dump(history.history, f, -1)
    
model.save_weights(model_path+'inceptionv3_ft_f.h5')
