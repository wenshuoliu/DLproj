import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--cohort', type=str, default='ami')
parser.add_argument('--tst_seed', type=int, default=0)
parser.add_argument('--val_seed', type=int, default=0)
parser.add_argument('--checkpoint_monitor', type=str, default='auc')
parser.add_argument('--n_avg', type=int, default=10)
parser.add_argument('--job_index', type=int, default=0)

args = parser.parse_args()
cohort = args.cohort
tst_seed = args.tst_seed
val_seed = args.val_seed
checkpoint_monitor = args.checkpoint_monitor
n_avg = args.n_avg
job_index = args.job_index

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import os, sys
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
from utils import Mat_reg

from ccs_tools import dx_multi, pr_multi, core_dtypes_pd
from setsum_layer import MaskedSum, MaskedDense

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
code_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE)*2 + sorted(pr_multi.ICD9CM_CODE) + sorted(dx_ccs_cat)[1:]*2 + sorted(pr_ccs_cat)[1:]
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

tst_key = pd.read_csv(path+'cohorts30/{}/tst_key{}.csv'.format(cohort, tst_seed), names = ['KEY_NRD'])
tst_df = all_df.loc[all_df.KEY_NRD.isin(tst_key.KEY_NRD)]
train_df0 = all_df.loc[~all_df.KEY_NRD.isin(tst_key.KEY_NRD)].reset_index()

age_mean = train_df0['AGE'].mean()
age_std = train_df0['AGE'].std()
los_mean = train_df0['LOS'].mean()
los_std = train_df0['LOS'].std()

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
pay1_mat_tst = to_categorical(tst_df.PAY1.values)[:, 1:]
los_array_tst = (tst_df.LOS.values - los_mean)/los_std
ed_mat_tst = to_categorical(tst_df.HCUP_ED.values)
zipinc_mat_tst = to_categorical(tst_df.ZIPINC_QRTL.values)[:, 1:]
#transfer_mat_tst = to_categorical(tst_df.SAMEDAYEVENT.values)
other_mat_tst = np.concatenate((demo_mat_tst, pay1_mat_tst, los_array_tst.reshape(los_array_tst.shape+(1,)), 
                                ed_mat_tst, zipinc_mat_tst), axis=1)
y_true = tst_df.readm30.astype(int).values

val_folds = 9
code_embed_dim = 200
hosp_embed_dim = 1
lr1 = 2e-4
lr2 = 2e-5
fc_width = 512
md_width = 128
dropout = 0.3
batchsize = 256
penalty = 0.
    
embed_glove = np.load(path+'all/sepdx1/test_embed/cosine/embed_mat_{0}_{1:.3f}_{2}_{3}{4}.npy'.format(code_embed_dim, 0, 20, cohort, 
                                                                                    tst_seed))
embed_initializer = Constant(embed_glove)

skf = StratifiedKFold(n_splits=val_folds, random_state=24, shuffle=True)

val_ind = 0
for trn_idx, val_idx in skf.split(train_df0, train_df0.HOSP_NRD):
    if not val_ind==val_seed:
        val_ind += 1
        continue
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
    
    demo_mat = train_df[['AGE', 'FEMALE']].values
    demo_mat[:, 0] = (demo_mat[:, 0]-age_mean)/age_std
    pay1_mat = to_categorical(train_df.PAY1.values)[:, 1:]
    los_array = train_df.LOS.values
    los_array = (los_array - los_mean)/los_std
    ed_mat = to_categorical(train_df.HCUP_ED.values)
    zipinc_mat = to_categorical(train_df.ZIPINC_QRTL.values)[:, 1:]
    #transfer_mat = to_categorical(train_df.SAMEDAYEVENT.values)
    other_mat = np.concatenate((demo_mat, pay1_mat, los_array.reshape(los_array.shape+(1,)), ed_mat, zipinc_mat), axis=1)
    other_mat_trn = other_mat[:N_trn, ]
    other_mat_val = other_mat[N_trn:, ]
    
    y = train_df['readm30'].values.astype(int)
    y_trn = y[:N_trn]
    y_val = y[N_trn:]
    
    y_pred_lst = []
    auc_lst = []
    for avg_ind in range(n_avg):
        # model building 
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
        input_hosp = Input(shape=(1,))
        hosp_embed = Embedding(input_dim=len(hosp_cat), output_dim=hosp_embed_dim, input_length=1)(input_hosp)
        hosp_embed = Reshape((hosp_embed_dim, ))(hosp_embed)
        input_other = Input(shape=(other_mat.shape[1], ))
        merged = Concatenate(axis=1)([DX1_embed, DX_embed, PR_embed, hosp_embed, input_other])
        x = Dense(fc_width, activation='relu')(merged)
        x = Dropout(dropout)(x)
        prediction = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[input_DX1, input_DX, input_PR, input_hosp, input_other], outputs=prediction)     
    
        for l in model.layers:
            if l.name=='DX_embed' or l.name=='PR_embed' or l.name=='DX1_embed':
                l.trainable = False
            
        adam = Adam(lr=lr1)
        model.compile(optimizer=adam, loss='binary_crossentropy')
    
        if checkpoint_monitor=='auc':
            checkpoint = AUCCheckPoint(filepath=model_path+'ami_glove_auc_temp1_'+str(job_index)+'.h5', validation_y=y_val, 
                                 validation_x=[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val])
        elif checkpoint_monitor=='valloss':
            checkpoint = ModelCheckpoint(filepath=model_path+'ami_glove_auc_temp1_'+str(job_index)+'.h5', save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=K.epsilon())
        earlystop = EarlyStopping(monitor='val_loss', patience=20)
    
        hist = model.fit([DX1_array_trn, DX_mat_trn, PR_mat_trn, hosp_array_trn, other_mat_trn], y_trn, 
                     batch_size=batchsize, epochs=100, callbacks=[checkpoint, reduce_lr, earlystop], 
                     validation_data=[[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val], y_val], 
                    verbose=2)
    
        model.load_weights(model_path+'ami_glove_auc_temp1_'+str(job_index)+'.h5')
    
        for l in model.layers:
            if l.name=='DX_embed' or l.name=='PR_embed' or l.name=='DX1_embed':
                l.trainable = True
        adam = Adam(lr=lr2)
        model.compile(optimizer=adam, loss='binary_crossentropy')
    
        if checkpoint_monitor=='auc':
            checkpoint = AUCCheckPoint(filepath=model_path+'ami_glove_auc_temp2_'+str(job_index)+'.h5', validation_y=y_val, 
                                 validation_x=[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val])
        elif checkpoint_monitor=='valloss':
            checkpoint = ModelCheckpoint(filepath=model_path+'ami_glove_auc_temp2_'+str(job_index)+'.h5', save_best_only=True, save_weights_only=True)
        hist = model.fit([DX1_array_trn, DX_mat_trn, PR_mat_trn, hosp_array_trn, other_mat_trn], y_trn, 
                     batch_size=batchsize, epochs=40, callbacks=[checkpoint, reduce_lr, earlystop], 
                     validation_data=[[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val], y_val], 
                    verbose=2)
    
        model.load_weights(model_path+'ami_glove_auc_temp2_'+str(job_index)+'.h5')
    
        y_pred = model.predict([DX1_array_tst, DX_mat_tst, PR_mat_tst, hosp_array_tst, other_mat_tst], verbose=0)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        model.save_weights(model_path+'testing/avg_{}_{}{}{}_{}.h5'.format(checkpoint_monitor, cohort, tst_seed, val_seed, avg_ind))
        auc_lst.append(roc_auc)
        y_pred_lst.append(y_pred)
        del(model)
    
    y_pred_mat = np.column_stack(y_pred_lst)
    y_pred_df = pd.DataFrame(y_pred_mat, columns=['pred'+str(j) for j in range(n_avg)])
    y_pred_df = y_pred_df.assign(pred_avg=y_pred_df.mean(axis=1))
    y_pred_df.to_csv(path+'cohorts30/{}/model_avg/pred_{}{}{}.csv'.format(cohort, checkpoint_monitor, tst_seed, val_seed), index=False)
    
    y_pred_avg = y_pred_df.pred_avg.values
    fpr, tpr, _ = roc_curve(y_true, y_pred_avg)
    auc_avg = auc(fpr, tpr)
    auc_lst.append(auc_avg)
    auc_df = pd.DataFrame(np.array(auc_lst).reshape((1,n_avg+1)), columns=['auc'+str(j) for j in range(n_avg)]+['auc_avg'])
    auc_df.to_csv(path+'cohorts30/{}/model_avg/auc_{}{}{}.csv'.format(cohort, checkpoint_monitor, tst_seed, val_seed), index=False)
    val_seed += 1

    
    
