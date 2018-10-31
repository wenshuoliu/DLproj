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
from utils import get_frequency, preprocess

from ccs_tools import dx_multi, pr_multi, core_dtypes_pd

all_df = pd.read_csv(path+'cohorts30/{}/pred_comorb.csv'.format(cohort), dtype=core_dtypes_pd)
tst_key = pd.read_csv(path+'cohorts30/{}/tst_key{}.csv'.format(cohort, tst_seed), names = ['KEY_NRD'])
tst_df = all_df.loc[all_df.KEY_NRD.isin(tst_key.KEY_NRD)]
train_df0 = all_df.loc[~all_df.KEY_NRD.isin(tst_key.KEY_NRD)].reset_index()

if eval_data=='index':
    index_df = pd.read_csv(path+'cohorts30/{}/index_comorb.csv'.format(cohort), dtype=core_dtypes_pd)
elif eval_data=='tst':
    index_df = tst_df.copy()  
    
#index_df = index_df.groupby('HOSP_NRD').apply(lambda x:x.sample(frac=resample_frac))
index_df = index_df.sample(frac=resample_frac)
index_df = index_df.reset_index(drop=True)

n_DX = 29
n_PR = 15
DXs = ['DX'+str(n) for n in range(2, n_DX+2)]
PRs = ['PR'+str(n) for n in range(1, n_PR+1)]

DX_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE)
PR_cat = ['missing'] + sorted(pr_multi.ICD9CM_CODE)
n_DX_cat = len(DX_cat)
n_PR_cat = len(PR_cat)
unclassified = set(dx_multi.loc[dx_multi.CCS_LVL1 == '18', 'ICD9CM_CODE'])
dx_ccs_cat = pd.concat([dx_multi.CCS_LVL1, dx_multi.CCS_LVL2, dx_multi.CCS_LVL3, dx_multi.CCS_LVL4]).astype('category').cat.categories
pr_ccs_cat = pd.concat([pr_multi.CCS_LVL1, pr_multi.CCS_LVL2, pr_multi.CCS_LVL3]).astype('category').cat.categories
DX1_dict = dict(zip(DX_cat, range(len(DX_cat))))
DX_dict = dict(zip(DX_cat, [0] + list(range(len(DX_cat), len(DX_cat)*2))))
PR_dict = dict(zip(PR_cat, [0] + list(range(len(DX_cat)*2-1, len(DX_cat)*2+len(PR_cat)-1))))
code_cat = ['missing']+sorted(dx_multi.ICD9CM_CODE)*2+sorted(pr_multi.ICD9CM_CODE)+sorted(dx_ccs_cat)[1:]*2+sorted(pr_ccs_cat)[1:]
n_code_cat = len(code_cat)

hosp_series = all_df['HOSP_NRD'].astype('category')
hosp_cat = hosp_series.cat.categories
hosp_dict = dict(zip(hosp_cat, range(len(hosp_cat))))

age_mean = train_df0['AGE'].mean()
age_std = train_df0['AGE'].std()
los_mean = train_df0['LOS'].mean()
los_std = train_df0['LOS'].std()
n_pay1 = int(train_df0['PAY1'].max())+1
n_ed = int(train_df0['HCUP_ED'].max())+1
n_zipinc = int(train_df0['ZIPINC_QRTL'].max())+1

DX1_array = index_df.DX1.map(DX1_dict).values
DX_df = index_df[DXs]
DX_df = DX_df.fillna('missing')
DX_df[DX_df.isin(['invl', 'incn'])] = 'missing'
DX_df[DX_df.isin(unclassified)] = 'missing'
for dx in DXs:
    DX_df[dx] = DX_df[dx].map(DX_dict)
DX_mat = DX_df.values
PR_df = index_df[PRs]
PR_df = PR_df.fillna('missing')
PR_df[PR_df.isin(['invl', 'incn'])] = 'missing'
for pr in PRs:
    PR_df[pr] = PR_df[pr].map(PR_dict)
PR_mat = PR_df.values
demo_mat = index_df[['AGE', 'FEMALE']].values
demo_mat[:, 0] = (demo_mat[:, 0]-age_mean)/age_std
hosp_array = index_df['HOSP_NRD'].map(hosp_dict).values
pay1_mat = to_categorical(index_df.PAY1.values, num_classes=n_pay1)[:, 1:]
los_array = (index_df.LOS.values - los_mean)/los_std
ed_mat = to_categorical(index_df.HCUP_ED.values, num_classes=n_ed)
zipinc_mat = to_categorical(index_df.ZIPINC_QRTL.values, num_classes=n_zipinc)[:, 1:]
#transfer_mat = to_categorical(index_df.SAMEDAYEVENT.values)
other_mat = np.concatenate((demo_mat, pay1_mat, los_array.reshape(los_array.shape+(1,)), 
                                ed_mat, zipinc_mat), axis=1)

from setsum_layer import SetSum, MaskedSum, MaskedDense, MaskedPooling

code_embed_dim = 200
md_width = 128
fc_width = 512
hosp_embed_dim = 1
dropout = 0.3

input_DX1 = Input(shape=(1,))
DX1_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, name='DX1_embed')(input_DX1)
DX1_embed = Reshape((code_embed_dim,))(DX1_embed)
input_DX = Input(shape = (n_DX,))
DX_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, mask_zero=True, name='DX_embed')(input_DX)
DX_embed = MaskedDense(md_width, activation='relu')(DX_embed)
DX_embed = MaskedSum()(DX_embed)
input_PR = Input(shape = (n_PR,))
PR_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, mask_zero=True, name='PR_embed')(input_PR)
PR_embed = MaskedDense(md_width, activation='relu')(PR_embed)
PR_embed = MaskedSum()(PR_embed)
input_hosp = Input(shape=(1,))
hosp_embed = Embedding(input_dim=len(hosp_cat), output_dim=hosp_embed_dim, input_length=1)(input_hosp)
hosp_embed = Reshape((hosp_embed_dim, ))(hosp_embed)
input_other = Input(shape=(other_mat.shape[1], ))
merged = Concatenate(axis=1)([DX1_embed, DX_embed, PR_embed, hosp_embed, input_other])
x = Dense(fc_width, activation='relu')(merged)
x = Dropout(dropout)(x)
prediction = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[input_DX1, input_DX, input_PR, input_hosp, input_other], outputs=prediction)

val_folds = 9
recycle_pred = np.zeros((len(hosp_cat), val_folds))
for val_seed in range(val_folds):
    model.load_weights(model_path+'best30_{}{}_{}.h5'.format(cohort, tst_seed, val_seed))
    #y_pred = model.predict([DX1_array, DX_mat, PR_mat, hosp_array, other_mat], verbose=0)
    #recycle_pred.append(y_pred.mean())
    for i, hosp in enumerate(hosp_cat):
        hosp_array = np.repeat(hosp_dict[hosp], len(index_df))
        y_pred = model.predict([DX1_array, DX_mat, PR_mat, hosp_array, other_mat], verbose=0)
        recycle_pred[i, val_seed] = y_pred.mean()
res_df = pd.DataFrame(recycle_pred, columns=['recycle_pred'+str(j) for j in range(val_folds)])
res_df = res_df.assign(HOSP_NRD=hosp_cat)
res_df.to_csv(path+'cohorts30/{}/recyc_pred_subsample_{}{}_{:.2E}.csv'.format(cohort, eval_data, tst_seed, resample_frac), index=False)
