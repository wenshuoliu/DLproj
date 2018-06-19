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
from sklearn.metrics import cohen_kappa_score, roc_curve, auc

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from keras_addon import ImageFrameGenerator, AUCCheckPoint

path  =  "/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/colonoscopy2/"
model_path = path + 'models/cross_validation/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)
data_path = path+'subset/'

batch_size=32

video_labels = pd.read_csv(path+'video_labels.csv')

from keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)

output1 = Dense(2, activation='softmax', name='split0_123')(x)
output2 = Dense(2, activation='softmax', name='split01_23')(x)
output3 = Dense(2, activation='softmax', name='split012_3')(x)

model = Model(inputs=base_model.input, outputs=[output1, output2, output3])

tst_gen = ImageFrameGenerator()
tst_itr = tst_gen.flow_from_frame(path+'all_video_images/', video_labels, 'filename', 
                                   ['Mayo_score'], label_types=['categorical'], 
                                     target_size=(256, 320), batch_size=batch_size, shuffle=False)

auc01_23 = []
acc01 = []
acc23 = []
acc0 = []
acc1 = []
acc2 = []
acc3 = []
kappa_linear = []
kappa_quad = []

for cv in range(1, 11):
    model.load_weights(model_path+'binary3_valloss'+str(cv)+'.h5')
    y = model.predict_generator(tst_itr, verbose=2)

    y_pred = np.concatenate([l[:, 1].reshape((l.shape[0], 1)) for l in y], axis=1)
    y_true = video_labels[['split0_123', 'split01_23', 'split012_3']].values

    fpr, tpr, _ = roc_curve(y_true[:, 1], y_pred[:, 1])
    auc01_23.append(auc(fpr, tpr))
    
    acc01.append(sum((y_pred[:, 1]<0.5) & (y_true[:, 1]==0))/sum(y_true[:, 1]==0))
    acc23.append(sum((y_pred[:, 1]>0.5) & (y_true[:, 1]==1))/sum(y_true[:, 1]==1))
    
    score4 = np.zeros((y_pred.shape[0], y_pred.shape[1]+1), dtype=float)
    score4[:, 0] = 1 - y_pred[:, 0]
    score4[:, 1] = y_pred[:, 0]*(1-y_pred[:,1])
    score4[:, 2] = y_pred[:, 0]*y_pred[:,1]*(1-y_pred[:,2])
    score4[:, 3] = y_pred[:, 0]*y_pred[:,1]*y_pred[:,2]

    mayo_pred = pd.Series(score4.argmax(axis=1), name='Mayo_pred')
    mayo_true = video_labels.Mayo_score
    mayo_true.name = 'Mayo_true'
    
    acc0.append(sum((mayo_pred==0) & (mayo_true==0))/sum(mayo_true==0))
    acc1.append(sum((mayo_pred==1) & (mayo_true==1))/sum(mayo_true==1))
    acc2.append(sum((mayo_pred==2) & (mayo_true==2))/sum(mayo_true==2))
    acc3.append(sum((mayo_pred==3) & (mayo_true==3))/sum(mayo_true==3))
    kappa_linear.append(cohen_kappa_score(mayo_pred.values, mayo_true.values, weights='linear'))
    kappa_quad.append(cohen_kappa_score(mayo_pred.values, mayo_true.values, weights='quadratic'))

np.savez('output/cv_video.csv', auc01_23=auc01_23, acc01=acc01, acc23=acc23, acc0=acc0, acc1=acc1, acc2=acc2, acc3=acc3, kappa_l=kappa_linear, kappa_q=kappa_quad)
