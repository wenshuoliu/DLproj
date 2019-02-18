import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--cohort', type=str, default='ami')
parser.add_argument('--tst_seed', type=int, default=0)
parser.add_argument('--eval_data', type=str, default='index', help='index means do recycled prediction on all index admissions, tst mean only on test set.')
parser.add_argument('--resample_frac', type=float, default=0.1)

parser.add_argument('--job_index', type=int, default=0)

args = parser.parse_args()
cohort = args.cohort
tst_seed = args.tst_seed
eval_data = args.eval_data
resample_frac = args.resample_frac

job_index = args.job_index

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import os, sys, time
from importlib import reload

path = '/nfs/turbo/umms-awaljee/wsliu/Data/NRD/'

model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
from keras.layers import Input, Embedding, Concatenate, Reshape, BatchNormalization, LSTM, CuDNNLSTM, CuDNNGRU, Lambda
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
from utils import get_frequency, preprocess_ms

from ccs_tools import dx_multi, pr_multi, core_dtypes_pd

all_df = pd.read_csv(path+'cohorts20/{}/pred_comorb.csv'.format(cohort), dtype=core_dtypes_pd)
tst_key = pd.read_csv(path+'cohorts20/{}/tst_key{}.csv'.format(cohort, tst_seed), names = ['KEY_NRD'])
tst_df = all_df.loc[all_df.KEY_NRD.isin(tst_key.KEY_NRD)]
train_df0 = all_df.loc[~all_df.KEY_NRD.isin(tst_key.KEY_NRD)].reset_index()

if eval_data=='index':
    index_df = pd.read_csv(path+'cohorts20/{}/index_comorb.csv'.format(cohort), dtype=core_dtypes_pd)
elif eval_data=='tst':
    index_df = tst_df.copy()  
    
#index_df = index_df.groupby('HOSP_NRD').apply(lambda x:x.sample(frac=resample_frac))
index_df = index_df.sample(frac=resample_frac)
index_df = index_df.reset_index(drop=True)

n_DX = 29
n_PR = 15
DXs = ['DX'+str(n) for n in range(2, n_DX+2)]
PRs = ['PR'+str(n) for n in range(1, n_PR+1)]

DX1_rarecutpoint = 0
DX_rarecutpoint = 0
PR_rarecutpoint = 0

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

preprocessed_ms = preprocess_ms(index_df, DX1_cat=DX1_cat, DX_cat=DX_cat, PR_cat=PR_cat)
dx1_ccs_dict = preprocessed_ms['dx1_ccs_dict']
dx_ccs_dict = preprocessed_ms['dx_ccs_dict']
pr_ccs_dict = preprocessed_ms['pr_ccs_dict']
dx1_parent_pairs = preprocessed_ms['dx1_parent_pairs']
dx_parent_pairs = preprocessed_ms['dx_parent_pairs']
pr_parent_pairs = preprocessed_ms['pr_parent_pairs']
hosp_cat = preprocessed_ms['hosp_cat']
hosp_dict = preprocessed_ms['hosp_dict']

age_mean = train_df0['AGE'].mean()
age_std = train_df0['AGE'].std()
los_mean = train_df0['LOS'].mean()
los_std = train_df0['LOS'].std()
n_pay1 = int(train_df0['PAY1'].max())+1
n_ed = int(train_df0['HCUP_ED'].max())+1
n_zipinc = int(train_df0['ZIPINC_QRTL'].max())+1

index_df = preprocessed_ms['int_df']
DX1_array = index_df.DX1.values
DX1_array = index_df['DX1'].values
DX_mat = index_df[DXs].values
PR_mat = index_df[PRs].values
hosp_array = index_df['HOSP_NRD'].values

demo_mat = index_df[['AGE', 'FEMALE']].values
demo_mat[:, 0] = (demo_mat[:, 0]-age_mean)/age_std
pay1_mat = to_categorical(index_df.PAY1.values, num_classes=n_pay1)[:, 1:]
los_array = (index_df.LOS.values - los_mean)/los_std
ed_mat = to_categorical(index_df.HCUP_ED.values, num_classes=n_ed)
zipinc_mat = to_categorical(index_df.ZIPINC_QRTL.values, num_classes=n_zipinc)[:, 1:]
#other_pred==0
other_mat = demo_mat


from setsum_layer import SetSum, MaskedSum, MaskedDense, MaskedPooling

DX1_dim = 200
DX_dim = 200
PR_dim = 100
fc_width = 512
hosp_embed_dim = 1
dropout = 0.3

input_DX1 = Input(shape=(1,))
DX1_embed = Embedding(input_dim=n_DX1_cat+len(dx1_ccs_dict)+1, output_dim=DX1_dim, 
                      name='DX1_embed')(input_DX1)     
DX1_embed = Reshape((DX1_dim,))(DX1_embed)
input_DX = Input(shape = (len(DXs),))
DX_embed = Embedding(input_dim=n_DX_cat+len(dx_ccs_dict)+1, output_dim=DX_dim, mask_zero=True,
                             name='DX_embed')(input_DX)
DX_embed = MaskedDense(DX_dim, activation='relu')(DX_embed)
DX_embed = MaskedSum()(DX_embed)
input_PR = Input(shape = (len(PRs),))
PR_embed = Embedding(input_dim=n_PR_cat+len(pr_ccs_dict)+1, output_dim=PR_dim, mask_zero=True, 
                             name='PR_embed')(input_PR)
PR_embed = MaskedDense(PR_dim, activation='relu')(PR_embed)
PR_embed = MaskedSum()(PR_embed)
input_hosp = Input(shape=(1,))
hosp_embed = Embedding(input_dim=len(hosp_cat), output_dim=hosp_embed_dim, input_length=1)(input_hosp)
hosp_embed = Reshape((hosp_embed_dim, ))(hosp_embed)
input_other = Input(shape=(other_mat.shape[1], ))
merged = Concatenate(axis=1)([DX1_embed, DX_embed, PR_embed, hosp_embed, input_other])
merged = Dense(fc_width, activation='relu')(merged)
merged = Dropout(dropout)(merged)
prediction = Dense(2, activation='softmax')(merged)
model = Model(inputs=[input_DX1, input_DX, input_PR, input_hosp, input_other], outputs=prediction)   
# save all validation folds:
'''
recycle_pred = np.zeros((len(hosp_cat), 9))
for val_ind in range(9):
    model.load_weights(model_path+'cohorts20/ms_{}{}{}.h5'.format(cohort, tst_seed, val_ind))
    for i, hosp in enumerate(hosp_cat):
        hosp_array = np.repeat(hosp_dict[hosp], len(index_df))
        y_pred = model.predict([DX1_array, DX_mat, PR_mat, hosp_array, other_mat], verbose=0)
        recycle_pred[i, val_ind] = y_pred[:, 1].mean()
res_df = pd.DataFrame(recycle_pred, columns=['recyc_pred'+str(j) for j in range(9)])
res_df = res_df.assign(HOSP_NRD=hosp_cat)
'''
recycle_pred = np.zeros((len(hosp_cat), ))
model.load_weights(model_path+'cohorts20/ms_notest_{}{}.h5'.format(cohort, tst_seed))
for i, hosp in enumerate(hosp_cat):
    hosp_array = np.repeat(hosp_dict[hosp], len(index_df))
    y_pred = model.predict([DX1_array, DX_mat, PR_mat, hosp_array, other_mat], verbose=0)
    recycle_pred[i] = y_pred[:, 1].mean()
res_df = pd.DataFrame(dict(HOSP_NRD=hosp_cat, recyc_pred=recycle_pred))

res_df.to_csv(path+'cohorts20/{}/recyc_pred_ms_notest_{}{}_{:.2f}.csv'.format(cohort, eval_data, tst_seed, resample_frac), index=False)
