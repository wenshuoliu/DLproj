""" Tune the hyper-parameters. """
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--cohort', type=str, default='ami')
parser.add_argument('--tst_seed', type=int, default=0, help='the seed to split training/test data')
parser.add_argument('--val_fold', type=int, default=10, help='number of folds to split training/validation data')
parser.add_argument('--result_file', type=str, default='output/result.csv')
parser.add_argument('--dx_rarecutpoint', type=int, default=10)
parser.add_argument('--pr_rarecutpoint', type=int, default=10)

parser.add_argument('--job_index', type=int, default=0)

args = parser.parse_args()
lr = args.lr
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
    
from keras.utils import to_categorical

module_path = '/home/wsliu/Codes/DLproj'
if module_path not in sys.path:
    sys.path.append(module_path)
if module_path+'/NRD' not in sys.path:
    sys.path.append(module_path+'/NRD')
from DL_utils import plot_roc
from utils import Parent_reg, get_frequency, preprocess

from ccs_tools import dx_multi, pr_multi, core_dtypes_pd

n_DX = 29
n_PR = 15
DXs = ['DX'+str(n) for n in range(2, n_DX+2)]
PRs = ['PR'+str(n) for n in range(1, n_PR+1)]
    
folder = 'multi_space_glove/'
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
other_mat_tst = demo_mat_tst
y_true = tst_df.readm30.astype(int).values
X_tst = np.concatenate([code_ohe_tst, hosp_ohe_tst, other_mat_tst], axis=1)

train_df = train_df0.copy()
    
code_mat_train = train_df[['DX1']+DXs+PRs].values
code_ohe_train = np.zeros((len(train_df), len(code_cat)))
for i in range(code_mat_train.shape[0]):
    for j in range(code_mat_train.shape[1]):
        if not code_mat_train[i,j]==0:
            code_ohe_train[i, code_mat_train[i,j]] = 1.
    
hosp_array_train = train_df['HOSP_NRD'].values
hosp_ohe_train = np.zeros((len(train_df), len(hosp_cat)))
for j, hosp in enumerate(hosp_array_train):
    hosp_ohe_train[j, hosp] = 1.
    
demo_mat_train = train_df[['AGE', 'FEMALE']].values
demo_mat_train[:, 0] = (demo_mat_train[:, 0]-age_mean)/age_std
other_mat_train = demo_mat_train

X_train = np.concatenate([code_ohe_train, hosp_ohe_train, other_mat_train], axis=1)
y_train = train_df.readm30.astype(int).values
    
# model training and testing
from xgboost import XGBClassifier
model = XGBClassifier()
xgb = model.fit(X_train, y_train)
y_pred = xgb.predict_proba(X_tst)

fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
roc_auc = auc(fpr, tpr)
with open(result_file.format(job_index), 'a') as f:
    f.write('{:.3f}{}{}{}{}{}{:.4f}\n'.format(lr, cohort, tst_seed, n_fold, DX_rarecutpoint, PR_rarecutpoint, roc_auc))
    
