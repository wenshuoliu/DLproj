""" Tune the hyper-parameters. """
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='setsum')
parser.add_argument('--code_embed_dim', type=int, default=100)
parser.add_argument('--fc_width', type=int, default=64)
parser.add_argument('--md_width', type=int, default=128, help='masked dense layer width')
parser.add_argument('--lr1', type=float, default=0.0002)
parser.add_argument('--lr2', type=float, default=0.00002)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--embed_file', type=str, default='random')
parser.add_argument('--cohort', type=str, default='ami')
parser.add_argument('--sep_dx1', type=int, default=0, help='whether separate DX1 when defining dict')
parser.add_argument('--tst_seed', type=int, default=0, help='the seed to split training/test data')
parser.add_argument('--val_fold', type=int, default=10, help='number of folds to split training/validation data')
parser.add_argument('--result_file', type=str, default='output/result.csv')
parser.add_argument('--rho_width', type=int, default=32)

parser.add_argument('--job_index', type=int, default=0)

args = parser.parse_args()
model_name = args.model_name
code_embed_dim = args.code_embed_dim
hosp_embed_dim = 1
fc_width = args.fc_width
md_width = args.md_width
lr1 = args.lr1
lr2 = args.lr2
dropout = args.dropout
batchsize = args.batchsize
embed_file = args.embed_file
cohort = args.cohort
sep_dx1 = args.sep_dx1
tst_seed = args.tst_seed
n_fold = args.val_fold
result_file = args.result_file
rho_width = args.rho_width

job_index = args.job_index

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import os, sys
from datetime import datetime
import statsmodels.stats.api as sms

path = '/nfs/turbo/umms-awaljee/wsliu/Data/NRD/'

model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
from keras.layers import Input, Embedding, Concatenate, Reshape, Lambda, GlobalMaxPooling1D, BatchNormalization
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

from ccs_tools import dx_multi, pr_multi, core_dtypes_pd
from setsum_layer import SetSum, MaskedSum, MaskedDense, MaskedPooling
    
#all_df = pd.read_csv(path+'cohorts/{}/{}_pred.csv'.format(cohort, cohort), dtype=core_dtypes_pd)
all_df = pd.read_csv(path+'cohorts30/{}/pred_comorb.csv'.format(cohort), dtype=core_dtypes_pd)

#define dictionaries from codes to int
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
dx1_ccs_dict = dict(zip(dx_ccs_cat[1:], range(1+len(dx_multi)*2+len(pr_multi), len(dx_multi)*2+len(pr_multi)+len(dx_ccs_cat))))
dx_ccs_dict = dict(zip(dx_ccs_cat[1:], range(1+len(dx_multi)*2+len(pr_multi)+len(dx_ccs_cat[1:]), 
                                             1+len(dx_multi)*2+len(pr_multi)+len(dx_ccs_cat[1:])*2)))
pr_ccs_dict = dict(zip(pr_ccs_cat[1:], range(1+len(dx_multi)*2+len(pr_multi)+len(dx_ccs_cat[1:])*2, n_code_cat)))


hosp_series = all_df['HOSP_NRD'].astype('category')
hosp_cat = hosp_series.cat.categories
hosp_dict = dict(zip(hosp_cat, range(len(hosp_cat))))

n_DX = 29
n_PR = 15
DXs = ['DX'+str(j) for j in range(2, n_DX+2)]
PRs = ['PR'+str(j) for j in range(1, n_PR+1)]

#tst_key = pd.read_csv(path+'cohorts/{}/tst_key{}.csv'.format(cohort, tst_seed), names = ['KEY_NRD'])
tst_key = pd.read_csv(path+'cohorts30/{}/tst_key{}.csv'.format(cohort, tst_seed), names = ['KEY_NRD'])
tst_df = all_df.loc[all_df.KEY_NRD.isin(tst_key.KEY_NRD)]
train_df0 = all_df.loc[~all_df.KEY_NRD.isin(tst_key.KEY_NRD)].reset_index()

age_mean = train_df0['AGE'].mean()
age_std = train_df0['AGE'].std()
los_mean = train_df0['LOS'].mean()
los_std = train_df0['LOS'].std()
n_pay1 = int(train_df0['PAY1'].max())+1
n_ed = int(train_df0['HCUP_ED'].max())+1
n_zipinc = int(train_df0['ZIPINC_QRTL'].max())+1

DX1_array_tst = tst_df.DX1.map(DX1_dict).values
DX_df_tst = tst_df[DXs]
DX_df_tst = DX_df_tst.fillna('missing')
DX_df_tst[DX_df_tst.isin(['invl', 'incn'])] = 'missing'
DX_df_tst[DX_df_tst.isin(unclassified)] = 'missing'
for dx in DXs:
    DX_df_tst[dx] = DX_df_tst[dx].map(DX_dict)
DX_mat_tst = DX_df_tst.values
PR_df_tst = tst_df[PRs]
PR_df_tst = PR_df_tst.fillna('missing')
PR_df_tst[PR_df_tst.isin(['invl', 'incn'])] = 'missing'
for pr in PRs:
    PR_df_tst[pr] = PR_df_tst[pr].map(PR_dict)
PR_mat_tst = PR_df_tst.values
demo_mat_tst = tst_df[['AGE', 'FEMALE']].values
demo_mat_tst[:, 0] = (demo_mat_tst[:, 0]-age_mean)/age_std
hosp_array_tst = tst_df['HOSP_NRD'].map(hosp_dict).values
pay1_mat_tst = to_categorical(tst_df.PAY1.values, num_classes=n_pay1)[:, 1:]
los_array_tst = (tst_df.LOS.values - los_mean)/los_std
ed_mat_tst = to_categorical(tst_df.HCUP_ED.values, num_classes=n_ed)
zipinc_mat_tst = to_categorical(tst_df.ZIPINC_QRTL.values, num_classes=n_zipinc)[:, 1:]
#transfer_mat_tst = to_categorical(tst_df.SAMEDAYEVENT.values)
other_mat_tst = np.concatenate((pay1_mat_tst, los_array_tst.reshape(los_array_tst.shape+(1,)), 
                                ed_mat_tst, zipinc_mat_tst), axis=1)
y_true = tst_df.readm30.astype(int).values

if embed_file=='random':
    embed_initializer = 'uniform'
else:
    embed_glove = np.load(path+embed_file)
    embed_initializer = Constant(embed_glove)

#split trn/val data, do a n_fold validation
y_pred_lst = []
auc_lst = []
auc_freeze_lst = []
skf = StratifiedKFold(n_splits=n_fold, random_state=24, shuffle=True)
val_seed = 0
for trn_idx, val_idx in skf.split(train_df0, train_df0.HOSP_NRD):
    trn_df = train_df0.loc[trn_idx]
    val_df = train_df0.loc[val_idx]
    N_trn = len(trn_df)
    train_df = pd.concat([trn_df, val_df])
    
    DX1_series = train_df['DX1'].map(DX1_dict)
    DX1_array = DX1_series.values
    DX1_array_trn = DX1_array[:N_trn]
    DX1_array_val = DX1_array[N_trn:]
    
    DX_df = train_df[DXs]
    DX_df = DX_df.fillna('missing')
    DX_df[DX_df.isin(['invl', 'incn'])] = 'missing'
    DX_df[DX_df.isin(unclassified)] = 'missing'
    for dx in DXs:
        DX_df[dx] = DX_df[dx].map(DX_dict)
    DX_mat = DX_df.values
    DX_mat_trn = DX_mat[:N_trn, ]
    DX_mat_val = DX_mat[N_trn:, ]
    
    PR_df = train_df[PRs]
    PR_df = PR_df.fillna('missing')
    PR_df[PR_df.isin(['invl', 'incn'])] = 'missing'
    for pr in PRs:
        PR_df[pr] = PR_df[pr].map(PR_dict)
    PR_mat = PR_df.values
    PR_mat_trn = PR_mat[:N_trn, ]
    PR_mat_val = PR_mat[N_trn:, ]
    
    hosp_series = train_df['HOSP_NRD'].map(hosp_dict)
    hosp_array = hosp_series.values
    hosp_array_trn = hosp_array[:N_trn]
    hosp_array_val = hosp_array[N_trn:]
    if model_name=='permutate_hosp':
        hosp_array_trn = np.random.choice(hosp_array_trn, size=hosp_array_trn.shape[0], replace=False)
    
    demo_mat = train_df[['AGE', 'FEMALE']].values
    demo_mat[:, 0] = (demo_mat[:, 0]-age_mean)/age_std
    demo_mat_trn = demo_mat[:N_trn, ]
    demo_mat_val = demo_mat[N_trn:, ]
    pay1_mat = to_categorical(train_df.PAY1.values, num_classes=n_pay1)[:, 1:]
    los_array = train_df.LOS.values
    los_array = (los_array - los_mean)/los_std
    ed_mat = to_categorical(train_df.HCUP_ED.values, num_classes=n_ed)
    zipinc_mat = to_categorical(train_df.ZIPINC_QRTL.values, num_classes=n_zipinc)[:, 1:]
    #transfer_mat = to_categorical(train_df.SAMEDAYEVENT.values)
    other_mat = np.concatenate((pay1_mat, los_array.reshape(los_array.shape+(1,)), ed_mat, zipinc_mat), axis=1)
    other_mat_trn = other_mat[:N_trn, ]
    other_mat_val = other_mat[N_trn:, ]
    
    y = train_df['readm30'].values.astype(int)
    y_trn = y[:N_trn]
    y_val = y[N_trn:]
    Y_trn = to_categorical(y_trn)
    Y_val = to_categorical(y_val)
    
    # model building 
    if model_name=='demo':
        input_demo = Input(shape=(demo_mat.shape[1],))
        x = Dense(fc_width, activation='relu')(input_demo)
        x = Dropout(dropout)(x)
        prediction = Dense(2, activation='softmax', name='prediction')(x)
        model = Model(inputs=[input_demo], outputs=prediction)
        X_trn = [demo_mat_trn]
        X_val = [demo_mat_val]
        X_tst = [demo_mat_tst]
    elif model_name=='demo_dx1':
        input_DX1 = Input(shape=(1,))
        DX1_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, embeddings_initializer=embed_initializer, 
                             name='DX1_embed')(input_DX1)
        DX1_embed = Reshape((code_embed_dim,))(DX1_embed)
        input_demo = Input(shape=(demo_mat.shape[1], ))
        merged = Concatenate(axis=1)([DX1_embed, input_demo])
        x = Dense(fc_width, activation='relu')(merged)
        x = Dropout(dropout)(x)
        prediction = Dense(2, activation='softmax', name='prediction')(x)
        model = Model(inputs=[input_DX1, input_demo], outputs=prediction)
        X_trn = [DX1_array_trn, demo_mat_trn]
        X_val = [DX1_array_val, demo_mat_val]
        X_tst = [DX1_array_tst, demo_mat_tst]
    elif model_name=='demo_dx1_dx':
        input_DX1 = Input(shape=(1,))
        DX1_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, embeddings_initializer=embed_initializer, 
                             name='DX1_embed')(input_DX1)
        DX1_embed = Reshape((code_embed_dim,))(DX1_embed)
        input_DX = Input(shape = (n_DX,))
        DX_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, mask_zero=True, embeddings_initializer=embed_initializer, 
                             name='DX_embed')(input_DX)
        DX_embed = MaskedDense(md_width, activation='relu')(DX_embed)
        DX_embed = MaskedSum()(DX_embed)
        input_demo = Input(shape=(demo_mat.shape[1], ))
        merged = Concatenate(axis=1)([DX1_embed, DX_embed, input_demo])
        x = Dense(fc_width, activation='relu')(merged)
        x = Dropout(dropout)(x)
        prediction = Dense(2, activation='softmax', name='prediction')(x)
        model = Model(inputs=[input_DX1, input_DX, input_demo], outputs=prediction)
        X_trn = [DX1_array_trn, DX_mat_trn, demo_mat_trn]
        X_val = [DX1_array_val, DX_mat_val, demo_mat_val]
        X_tst = [DX1_array_tst, DX_mat_tst, demo_mat_tst]
    elif model_name=='demo_dx1_dx_pr':
        input_DX1 = Input(shape=(1,))
        DX1_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, embeddings_initializer=embed_initializer, 
                             name='DX1_embed')(input_DX1)
        DX1_embed = Reshape((code_embed_dim,))(DX1_embed)
        input_DX = Input(shape = (n_DX,))
        DX_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, mask_zero=True, embeddings_initializer=embed_initializer, 
                             name='DX_embed')(input_DX)
        DX_embed = MaskedDense(md_width, activation='relu')(DX_embed)
        DX_embed = MaskedSum()(DX_embed)
        input_PR = Input(shape = (n_PR,))
        PR_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, mask_zero=True, embeddings_initializer=embed_initializer,
                             name='PR_embed')(input_PR)
        PR_embed = MaskedDense(md_width, activation='relu')(PR_embed)
        PR_embed = MaskedSum()(PR_embed)
        input_demo = Input(shape=(demo_mat.shape[1], ))
        merged = Concatenate(axis=1)([DX1_embed, DX_embed, PR_embed, input_demo])
        x = Dense(fc_width, activation='relu')(merged)
        x = Dropout(dropout)(x)
        prediction = Dense(2, activation='softmax', name='prediction')(x)
        model = Model(inputs=[input_DX1, input_DX, input_PR, input_demo], outputs=prediction)
        X_trn = [DX1_array_trn, DX_mat_trn, PR_mat_trn, demo_mat_trn]
        X_val = [DX1_array_val, DX_mat_val, PR_mat_val, demo_mat_val]
        X_tst = [DX1_array_tst, DX_mat_tst, PR_mat_tst, demo_mat_tst]
          
    for l in model.layers:
        if l.name=='DX_embed' or l.name=='PR_embed' or l.name=='DX1_embed':
            l.trainable = False
            
    adam = Adam(lr=lr1)
    model.compile(optimizer=adam, loss='binary_crossentropy')
    
    auccheckpoint = AUCCheckPoint(filepath=model_path+'ami_glove_auc_temp1_'+str(job_index)+'.h5', validation_y=Y_val, 
                                 validation_x=X_val)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=K.epsilon())
    earlystop = EarlyStopping(monitor='val_loss', patience=20)
    
    class_weight = {0:(Y_trn.shape[0]/sum(Y_trn[:, 0])), 1:(Y_trn.shape[0]/sum(Y_trn[:, 1]))}
        
    hist = model.fit(X_trn, Y_trn, batch_size=batchsize, epochs=50, callbacks=[auccheckpoint, reduce_lr, earlystop], 
                     validation_data=[X_val, Y_val], verbose=2)
    
    model.load_weights(model_path+'ami_glove_auc_temp1_'+str(job_index)+'.h5')
    y = model.predict(X_tst, verbose=0)
    y_pred = y[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    auc_freeze_lst.append(roc_auc)
    
    for l in model.layers:
        if l.name=='DX_embed' or l.name=='PR_embed' or l.name=='DX1_embed':
            l.trainable = True
    adam = Adam(lr=lr2)
    model.compile(optimizer=adam, loss='binary_crossentropy')
    
    auccheckpoint = AUCCheckPoint(filepath=model_path+'ami_glove_auc_temp2_'+str(job_index)+'.h5', validation_y=Y_val, 
                                 validation_x=X_val)
    hist = model.fit(X_trn, Y_trn, batch_size=batchsize, epochs=20, callbacks=[auccheckpoint, reduce_lr, earlystop], 
                     validation_data=[X_val, Y_val], verbose=2)
    
    model.load_weights(model_path+'ami_glove_auc_temp2_'+str(job_index)+'.h5')
    
    y = model.predict(X_tst, verbose=0)
    y_pred = y[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    #model.save_weights(model_path+'best30_hosp_{}{}_{}.h5'.format(cohort, tst_seed, val_seed))
    auc_lst.append(roc_auc)
    y_pred_lst.append(y_pred)
    val_seed += 1

auc_mean = np.mean(auc_lst)
auc_freeze_mean = np.mean(auc_freeze_lst)
#auc_ci = '({0:.4f}, {1:.4f})'.format(*sms.DescrStatsW(auc_lst).tconfint_mean())
y_pred_mat = np.column_stack(y_pred_lst)
now = datetime.now().strftime('%y_%m_%d_%I_%M_%S')
y_pred_file = 'output/y_pred_mat'+now+'.npy'
np.save(y_pred_file, y_pred_mat)
y_pred_avg = y_pred_mat.mean(axis=1)
fpr, tpr, _ = roc_curve(y_true, y_pred_avg)
auc_avg = auc(fpr, tpr)
with open(result_file.format(job_index), 'a') as f:
    f.write('{},{},{},{},{},{:.1E},{:.1E},{:.1f},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{},{}\n'.format(model_name, code_embed_dim, hosp_embed_dim, fc_width, md_width, lr1, lr2, dropout, batchsize, embed_file, cohort, sep_dx1, tst_seed, n_fold, auc_mean, auc_avg, auc_freeze_mean, rho_width, y_pred_file))
    
