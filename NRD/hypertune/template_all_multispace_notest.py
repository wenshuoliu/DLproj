""" Tune the hyper-parameters. """
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='setsum_nn')
parser.add_argument('--DX1_dim', type=int, default=10)
parser.add_argument('--DX_dim', type=int, default=100)
parser.add_argument('--PR_dim', type=int, default=50)
parser.add_argument('--fc_width', type=int, default=64)
#parser.add_argument('--md_width', type=int, default=128, help='masked dense layer width')
parser.add_argument('--lr1', type=float, default=0.0002)
parser.add_argument('--lr2', type=float, default=0.00002)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--embed_file', type=str, default='random')
parser.add_argument('--cohort', type=str, default='ami')
parser.add_argument('--tst_seed', type=int, default=0, help='the seed to split training/test data')
parser.add_argument('--val_fold', type=int, default=10, help='number of folds to split training/validation data')
parser.add_argument('--result_file', type=str, default='output/result.csv')
parser.add_argument('--penalty', type=float, default=0.5)
parser.add_argument('--penalty_metric', type=str, default='l2')
parser.add_argument('--count_cap', type=int, default=100)
parser.add_argument('--dx1_rarecutpoint', type=int, default=10)
parser.add_argument('--dx_rarecutpoint', type=int, default=10)
parser.add_argument('--pr_rarecutpoint', type=int, default=10)
parser.add_argument('--other_pred', type=int, default=1, help='0 means no other predictors')
parser.add_argument('--ndxpr', type=int, default=0, help='1 means including numbers of dx/pr as predictors')

parser.add_argument('--job_index', type=int, default=0)

args = parser.parse_args()
model_name = args.model_name
DX1_dim = args.DX1_dim
DX_dim = args.DX_dim
PR_dim = args.PR_dim
hosp_embed_dim = 1
fc_width = args.fc_width
#md_width = args.md_width
lr1 = args.lr1
lr2 = args.lr2
dropout = args.dropout
batchsize = args.batchsize
embed_file = args.embed_file
cohort = args.cohort
tst_seed = args.tst_seed
n_fold = args.val_fold
result_file = args.result_file
penalty = args.penalty
penalty_metric = args.penalty_metric
count_cap = args.count_cap
DX1_rarecutpoint = args.dx1_rarecutpoint
DX_rarecutpoint = args.dx_rarecutpoint
PR_rarecutpoint = args.pr_rarecutpoint
other_pred = args.other_pred
ndxpr = args.ndxpr

job_index = args.job_index

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import os, sys
from datetime import datetime
import statsmodels.stats.api as sms

path = '/nfs/turbo/umms-awaljee/wsliu/Data/NRD/'

model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
from keras.layers import Input, Embedding, Concatenate, Reshape, Lambda, BatchNormalization
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
from utils import Parent_reg, get_frequency, preprocess, preprocess_ms

from ccs_tools import dx_multi, pr_multi, core_dtypes_pd
from setsum_layer import SetSum, MaskedSum, MaskedDense, MaskedPooling
from glove import GloveMS
    
unclassified = set(dx_multi.loc[dx_multi.CCS_LVL1 == '18', 'ICD9CM_CODE'])
DX1_freq = pd.read_csv(path+'all/DX1_freq.csv', dtype={'DX1':str, 'frequency':int})
DX_freq = pd.read_csv(path+'all/DX_freq.csv', dtype={'DX':str, 'frequency':int})
PR_freq = pd.read_csv(path+'all/PR_freq.csv', dtype={'PR':str, 'frequency':int})
DX1_freq = DX1_freq.loc[DX1_freq.frequency>0]
DX_freq = DX_freq.loc[DX_freq.frequency>0]
PR_freq = PR_freq.loc[PR_freq.frequency>0]
DX1_cat = sorted(DX1_freq.loc[(DX1_freq.frequency>DX1_rarecutpoint) & (~DX1_freq.DX1.isin(unclassified))].DX1)
DX_cat = sorted(DX_freq.loc[(DX_freq.frequency>DX_rarecutpoint) & (~DX_freq.DX.isin(unclassified))].DX)
PR_cat = sorted(PR_freq.loc[(PR_freq.frequency>PR_rarecutpoint)].PR)
n_DX1_cat = len(DX1_cat)
n_DX_cat = len(DX_cat)
n_PR_cat = len(PR_cat)

cooccur_all = pd.read_csv(path+'multi_space_glove/cooccur_df_all.csv')
all_df = pd.read_csv(path+'cohorts20/{}/pred_comorb.csv'.format(cohort), dtype=core_dtypes_pd)

preprocessed_ms = preprocess_ms(all_df, DX1_cat=DX1_cat, DX_cat=DX_cat, PR_cat=PR_cat)
dx1_ccs_dict = preprocessed_ms['dx1_ccs_dict']
dx_ccs_dict = preprocessed_ms['dx_ccs_dict']
pr_ccs_dict = preprocessed_ms['pr_ccs_dict']
dx1_parent_pairs = preprocessed_ms['dx1_parent_pairs']
dx_parent_pairs = preprocessed_ms['dx_parent_pairs']
pr_parent_pairs = preprocessed_ms['pr_parent_pairs']
hosp_cat = preprocessed_ms['hosp_cat']

gms = GloveMS(n_DX1_cat=n_DX1_cat, n_DX_cat=n_DX_cat, n_PR_cat=n_PR_cat, n_dx1_ccs=len(dx1_ccs_dict), 
                  n_dx_ccs=len(dx_ccs_dict), n_pr_ccs=len(pr_ccs_dict), DX1_dim=DX1_dim, DX_dim=DX_dim, PR_dim=PR_dim, 
                  count_cap=count_cap, scaling_factor=0.75)
gms.train_glove(cooccur_df=cooccur_all, cache_path=model_path+'temp/{}/'.format(job_index), dx1_parent_pairs=dx1_parent_pairs, 
                    dx_parent_pairs=dx_parent_pairs, pr_parent_pairs=pr_parent_pairs, lamb=penalty, metric=penalty_metric, verbose=2, 
                   batch_size=1024*8, epochs=40, earlystop_patience=10, reducelr_patience=2)
embeds = gms.get_embed_mat()

all_df = preprocessed_ms['int_df']
tst_key = pd.read_csv(path+'cohorts20/{}/tst_key{}.csv'.format(cohort, tst_seed), names = ['KEY_NRD'])
tst_df = all_df.loc[all_df.KEY_NRD.isin(tst_key.KEY_NRD)]
train_df0 = all_df.loc[~all_df.KEY_NRD.isin(tst_key.KEY_NRD)].reset_index()

## convert different variables into different np.array
n_DX = 29
n_PR = 15
DXs = ['DX'+str(j) for j in range(2, n_DX+2)]
PRs = ['PR'+str(j) for j in range(1, n_PR+1)]

age_mean = train_df0['AGE'].mean()
age_std = train_df0['AGE'].std()
los_mean = train_df0['LOS'].mean()
los_std = train_df0['LOS'].std()
n_pay1 = int(train_df0['PAY1'].max())+1
n_ed = int(train_df0['HCUP_ED'].max())+1
n_zipinc = int(train_df0['ZIPINC_QRTL'].max())+1

DX1_array_tst = tst_df['DX1'].values
DX_mat_tst = tst_df[DXs].values
PR_mat_tst = tst_df[PRs].values
hosp_array_tst = tst_df['HOSP_NRD'].values
    
demo_mat_tst = tst_df[['AGE', 'FEMALE']].values
demo_mat_tst[:, 0] = (demo_mat_tst[:, 0]-age_mean)/age_std
pay1_mat_tst = to_categorical(tst_df.PAY1.values, num_classes=n_pay1)[:, 1:]
los_array_tst = (tst_df.LOS.values - los_mean)/los_std
ed_mat_tst = to_categorical(tst_df.HCUP_ED.values, num_classes=n_ed)
zipinc_mat_tst = to_categorical(tst_df.ZIPINC_QRTL.values, num_classes=n_zipinc)[:, 1:]
#transfer_mat_tst = to_categorical(tst_df.SAMEDAYEVENT.values)
ncode_mat_tst = tst_df[['NDX', 'NPR']].values
if other_pred==0:
    other_mat_tst = demo_mat_tst
else:
    other_mat_tst = np.concatenate((demo_mat_tst, pay1_mat_tst, los_array_tst.reshape(los_array_tst.shape+(1,)), 
                                    ed_mat_tst, zipinc_mat_tst), axis=1)
if ndxpr==1:
    other_mat_tst = np.concatenate((other_mat_tst, ncode_mat_tst), axis=1)
y_true = tst_df.readm30.astype(int).values

train_df = train_df0.copy()
DX1_array_trn = train_df['DX1'].values
DX1_array_val = DX1_array_tst

DX_mat_trn = train_df[DXs].values
DX_mat_val = DX_mat_tst

PR_mat_trn = train_df[PRs].values
PR_mat_val = PR_mat_tst
    
hosp_array_trn = train_df['HOSP_NRD'].values
hosp_array_val = hosp_array_tst
    
demo_mat_train = train_df[['AGE', 'FEMALE']].values
demo_mat_train[:, 0] = (demo_mat_train[:, 0]-age_mean)/age_std
pay1_mat_train = to_categorical(train_df.PAY1.values, num_classes=n_pay1)[:, 1:]
los_array_train = (train_df.LOS.values - los_mean)/los_std
ed_mat_train = to_categorical(train_df.HCUP_ED.values, num_classes=n_ed)
zipinc_mat_train = to_categorical(train_df.ZIPINC_QRTL.values, num_classes=n_zipinc)[:, 1:]
ncode_mat_train = train_df[['NDX', 'NPR']].values
if other_pred==0:
    other_mat_train = demo_mat_train
else:
    other_mat_train = np.concatenate((demo_mat_train, pay1_mat_train, los_array_train.reshape(los_array_train.shape+(1,)), 
                                        ed_mat_train, zipinc_mat_train), axis=1)
if ndxpr==1:
    other_mat_train = np.concatenate((other_mat_train, ncode_mat_train), axis=1)
other_mat_trn = other_mat_train
other_mat_val = other_mat_tst

y_trn = train_df.readm30.astype(int).values
y_val = y_true
Y_trn = to_categorical(y_trn)
Y_val = to_categorical(y_val)
    
# model building 
if model_name=='setsum_nn':
    input_DX1 = Input(shape=(1,))
    DX1_embed = Embedding(input_dim=n_DX1_cat+len(dx1_ccs_dict)+1, output_dim=DX1_dim, 
                              embeddings_initializer=Constant(embeds['DX1_embed']), name='DX1_embed')(input_DX1)     
    DX1_embed = Reshape((DX1_dim,))(DX1_embed)
    input_DX = Input(shape = (len(DXs),))
    DX_embed = Embedding(input_dim=n_DX_cat+len(dx_ccs_dict)+1, output_dim=DX_dim, mask_zero=True, 
                             embeddings_initializer=Constant(embeds['DX_embed']), 
                             name='DX_embed')(input_DX)
    DX_embed = MaskedDense(DX_dim, activation='relu')(DX_embed)
    DX_embed = MaskedSum()(DX_embed)
    input_PR = Input(shape = (len(PRs),))
    PR_embed = Embedding(input_dim=n_PR_cat+len(pr_ccs_dict)+1, output_dim=PR_dim, mask_zero=True, 
                             embeddings_initializer=Constant(embeds['PR_embed']), 
                             name='PR_embed')(input_PR)
    PR_embed = MaskedDense(PR_dim, activation='relu')(PR_embed)
    PR_embed = MaskedSum()(PR_embed)
    input_hosp = Input(shape=(1,))
    hosp_embed = Embedding(input_dim=len(hosp_cat), output_dim=hosp_embed_dim, input_length=1)(input_hosp)
    hosp_embed = Reshape((hosp_embed_dim, ))(hosp_embed)
    input_other = Input(shape=(other_mat_train.shape[1], ))
    merged = Concatenate(axis=1)([DX1_embed, DX_embed, PR_embed, hosp_embed, input_other])
    merged = Dense(fc_width, activation='relu')(merged)
    merged = Dropout(dropout)(merged)
    prediction = Dense(2, activation='softmax')(merged)
    model = Model(inputs=[input_DX1, input_DX, input_PR, input_hosp, input_other], outputs=prediction)
   
for l in model.layers:
    if l.name=='DX_embed' or l.name=='PR_embed' or l.name=='DX1_embed':
        l.trainable = False
            
adam = Adam(lr=lr1)
model.compile(optimizer=adam, loss='categorical_crossentropy')
    
auccheckpoint = AUCCheckPoint(filepath=model_path+'embeding_nn_sub_temp1_'+str(job_index)+'.h5', validation_y=Y_val[:, 1], 
                                 validation_x=[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=K.epsilon())
earlystop = EarlyStopping(monitor='val_loss', patience=20)
    
#class_weight = {0:(Y_trn.shape[0]/sum(Y_trn[:, 0])), 1:(Y_trn.shape[0]/sum(Y_trn[:, 1]))}
class_weight = {0:1., 1:1.}
    
hist = model.fit([DX1_array_trn, DX_mat_trn, PR_mat_trn, hosp_array_trn, other_mat_trn], Y_trn, 
                     batch_size=batchsize, epochs=40, callbacks=[auccheckpoint, reduce_lr, earlystop], class_weight=class_weight, 
                     validation_data=[[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val], Y_val], 
                    verbose=2)
    
model.load_weights(model_path+'embeding_nn_sub_temp1_'+str(job_index)+'.h5')
    
for l in model.layers:
    if l.name=='DX_embed' or l.name=='PR_embed' or l.name=='DX1_embed':
        l.trainable = True
adam = Adam(lr=lr2)
model.compile(optimizer=adam, loss='categorical_crossentropy')
    
auccheckpoint = AUCCheckPoint(filepath=model_path+'embeding_nn_sub_temp2_'+str(job_index)+'.h5', validation_y=Y_val[:, 1], 
                                 validation_x=[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val])
hist = model.fit([DX1_array_trn, DX_mat_trn, PR_mat_trn, hosp_array_trn, other_mat_trn], Y_trn, 
                     batch_size=batchsize, epochs=20, callbacks=[auccheckpoint, reduce_lr, earlystop], class_weight=class_weight, 
                     validation_data=[[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val], Y_val], 
                    verbose=2)
    
model.load_weights(model_path+'embeding_nn_sub_temp2_'+str(job_index)+'.h5')
    
model.save_weights(model_path+'cohorts20/ms_notest_{}{}.h5'.format(cohort, tst_seed))