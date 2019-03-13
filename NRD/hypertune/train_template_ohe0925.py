""" Tune the hyper-parameters. """
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--fc_width1', type=int, default=1024)
parser.add_argument('--fc_width2', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--cohort', type=str, default='ami')
parser.add_argument('--tst_seed', type=int, default=0, help='the seed to split training/test data')
parser.add_argument('--val_fold', type=int, default=10, help='number of folds to split training/validation data')
parser.add_argument('--result_file', type=str, default='output/result.csv')
parser.add_argument('--dx_rarecutpoint', type=int, default=10)
parser.add_argument('--pr_rarecutpoint', type=int, default=10)

parser.add_argument('--job_index', type=int, default=0)

args = parser.parse_args()
fc_width1 = args.fc_width1
fc_width2 = args.fc_width2
lr = args.lr
dropout = args.dropout
batchsize = args.batchsize
cohort = args.cohort
tst_seed = args.tst_seed
n_fold = args.val_fold
result_file = args.result_file
DX_rarecutpoint = args.dx_rarecutpoint
PR_rarecutpoint = args.pr_rarecutpoint

job_index = args.job_index

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import os, sys
from datetime import datetime

path = '/nfs/turbo/umms-awaljee/wsliu/Data/NRD/'

model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
from keras.layers import Input, Embedding, Concatenate, Reshape, Lambda
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.initializers import Constant
import keras.backend as K

module_path = '/home/wsliu/Codes/DLproj'
if module_path not in sys.path:
    sys.path.append(module_path)
if module_path+'/NRD' not in sys.path:
    sys.path.append(module_path+'/NRD')
from DL_utils import plot_roc
from keras_addon import AUCCheckPoint
from utils import Parent_reg, get_frequency, preprocess

from ccs_tools import dx_multi, pr_multi, core_dtypes_pd
from setsum_layer import SetSum, MaskedSum, MaskedDense

n_DX = 29
n_PR = 15
DXs = ['DX'+str(n) for n in range(2, n_DX+2)]
PRs = ['PR'+str(n) for n in range(1, n_PR+1)]
    
folder = 'elder/'
all_df = pd.read_csv(path+folder+'cohorts10/{}/pred_comorb.csv'.format(cohort), dtype=core_dtypes_pd)
preprocessed = preprocess(all_df, DX_rarecutpoint=DX_rarecutpoint, PR_rarecutpoint=PR_rarecutpoint)
all_df = preprocessed['int_df']

tst_key = pd.read_csv(path+folder+'cohorts10/{}/tst_key{}.csv'.format(cohort, tst_seed), names = ['KEY_NRD'])
tst_df = all_df.loc[all_df.KEY_NRD.isin(tst_key.KEY_NRD)]
train_df0 = all_df.loc[~all_df.KEY_NRD.isin(tst_key.KEY_NRD)].reset_index()

## convert different variables into different np.array
DX1_cat = preprocessed['DX1_cat']
DX_cat = preprocessed['DX_cat']
PR_cat = preprocessed['PR_cat']
DX1_dict = preprocessed['DX1_dict']
DX_dict = preprocessed['DX_dict']
PR_dict = preprocessed['PR_dict']
code_cat = preprocessed['code_cat']
hosp_cat = preprocessed['hosp_cat']
age_mean = train_df0['AGE'].mean()
age_std = train_df0['AGE'].std()
#los_mean = train_df0['LOS'].mean()
#los_std = train_df0['LOS'].std()

code_mat_tst = tst_df[['DX1']+DXs+PRs].values
code_ohe_tst = np.zeros((len(tst_df), len(code_cat)))
for i in range(code_mat_tst.shape[0]):
    for j in range(code_mat_tst.shape[1]):
        if not code_mat_tst[i,j]==0:
            code_ohe_tst[i, code_mat_tst[i,j]] = 1.

hosp_array_tst = tst_df['HOSP_NRD'].values
hosp_ohe_tst = np.zeros((len(tst_df), len(hosp_cat)))
for j, hosp in enumerate(hosp_array_tst):
    hosp_ohe_tst[j, hosp] = 1.
    
demo_mat_tst = tst_df[['AGE', 'FEMALE']].values
demo_mat_tst[:, 0] = (demo_mat_tst[:, 0]-age_mean)/age_std
#pay1_mat_tst = to_categorical(tst_df.PAY1.values)[:, 1:]
#los_array_tst = (tst_df.LOS.values - los_mean)/los_std
#ed_mat_tst = to_categorical(tst_df.HCUP_ED.values)
#zipinc_mat_tst = to_categorical(tst_df.ZIPINC_QRTL.values)[:, 1:]
#transfer_mat_tst = to_categorical(tst_df.SAMEDAYEVENT.values)
#other_pred==0
other_mat_tst = demo_mat_tst
#other_pred==1
#other_mat_tst = np.concatenate((demo_mat_tst, pay1_mat_tst, los_array_tst.reshape(los_array_tst.shape+(1,)), 
#                                ed_mat_tst, zipinc_mat_tst, transfer_mat_tst), axis=1)
y_true = tst_df.readm30.astype(int).values

#split trn/val data, do a n_fold validation
y_pred_lst = []
auc_lst = []
skf = StratifiedKFold(n_splits=n_fold, random_state=24, shuffle=True)
for trn_idx, val_idx in skf.split(train_df0, train_df0.HOSP_NRD):
    train_df = train_df0.copy()
    
    code_mat_train = train_df[['DX1']+DXs+PRs].values
    code_ohe_train = np.zeros((len(train_df), len(code_cat)))
    for i in range(code_mat_train.shape[0]):
        for j in range(code_mat_train.shape[1]):
            if not code_mat_train[i,j]==0:
                code_ohe_train[i, code_mat_train[i,j]] = 1.
            
    code_ohe_trn = code_ohe_train[trn_idx, :]
    code_ohe_val = code_ohe_train[val_idx, :]
    
    hosp_array_train = train_df['HOSP_NRD'].values
    hosp_ohe_train = np.zeros((len(train_df), len(hosp_cat)))
    for j, hosp in enumerate(hosp_array_train):
        hosp_ohe_train[j, hosp] = 1.
    
    hosp_ohe_trn = hosp_ohe_train[trn_idx, :]
    hosp_ohe_val = hosp_ohe_train[val_idx, :]
    
    demo_mat_train = train_df[['AGE', 'FEMALE']].values
    demo_mat_train[:, 0] = (demo_mat_train[:, 0]-age_mean)/age_std
    #pay1_mat_train = to_categorical(train_df.PAY1.values)[:, 1:]
    #los_array_train = (train_df.LOS.values - los_mean)/los_std
    #ed_mat_train = to_categorical(train_df.HCUP_ED.values)
    #zipinc_mat_train = to_categorical(train_df.ZIPINC_QRTL.values)[:, 1:]
    #transfer_mat_train = to_categorical(train_df.SAMEDAYEVENT.values)
    other_mat_train = demo_mat_train
    #other_mat_train = np.concatenate((demo_mat_train, pay1_mat_train, los_array_train.reshape(los_array_train.shape+(1,)), 
    #                                ed_mat_train, zipinc_mat_train, transfer_mat_train), axis=1)

    other_mat_trn = other_mat_train[trn_idx, :]
    other_mat_val = other_mat_train[val_idx, :]

    y_train = train_df.readm30.astype(int).values
    Y_trn = to_categorical(y_train[trn_idx])
    Y_val = to_categorical(y_train[val_idx])
    
    # model building 
    input_code = Input(shape = (len(code_cat),))
    input_hosp = Input(shape=(len(hosp_cat),))
    input_other = Input(shape=(other_mat_train.shape[1], ))
    merged = Concatenate(axis=1)([input_code, input_hosp, input_other])
    x = Dense(fc_width1, activation='relu')(merged)
    x = Dense(fc_width2, activation='relu')(x)
    x = Dropout(dropout)(x)
    prediction = Dense(2, activation='softmax')(x)
    model = Model(inputs=[input_code, input_hosp, input_other], outputs=prediction)
    
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    
    auccheckpoint = AUCCheckPoint(filepath=model_path+'ohe_temp1_'+str(job_index)+'.h5', validation_y=Y_val[:, 1], 
                                validation_x=[code_ohe_val, hosp_ohe_val, other_mat_val])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=K.epsilon())
    earlystop = EarlyStopping(monitor='val_loss', patience=10)
    
    #class_weight = {0:(Y_trn.shape[0]/sum(Y_trn[:, 0])), 1:(Y_trn.shape[0]/sum(Y_trn[:, 1]))}
    class_weight = {0:1., 1:1.}
    
    hist = model.fit([code_ohe_trn, hosp_ohe_trn, other_mat_trn], Y_trn, 
                     batch_size=batchsize, epochs=10, callbacks=[auccheckpoint, reduce_lr, earlystop], class_weight=class_weight, 
                     validation_data=[[code_ohe_val, hosp_ohe_val, other_mat_val], Y_val], verbose=2)
    
    model.load_weights(model_path+'ohe_temp1_'+str(job_index)+'.h5')
    y_output = model.predict([code_ohe_tst, hosp_ohe_tst, other_mat_tst], verbose=0)  
    y_pred = y_output[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    auc_lst.append(roc_auc)
    y_pred_lst.append(y_pred)

auc_mean = np.mean(auc_lst)
y_pred_mat = np.column_stack(y_pred_lst)
now = datetime.now().strftime('%y_%m_%d_%I_%M_%S')
y_pred_file = path+'y_pred_mat/y_pred_mat'+now+'.npy'
np.save(y_pred_file, y_pred_mat)
y_pred_avg = y_pred_mat.mean(axis=1)
fpr, tpr, _ = roc_curve(y_true, y_pred_avg)
auc_avg = auc(fpr, tpr)
with open(result_file.format(job_index), 'a') as f:
    f.write('{0},{1},{2:.1E},{3:.1f},{4},{5},{6},{7},{8},{9},{10:.5f},{11:.5f},{12}\n'.format(fc_width1, fc_width2, lr, dropout, batchsize, cohort, tst_seed, n_fold, DX_rarecutpoint, PR_rarecutpoint, auc_mean, auc_avg, y_pred_file))
    
