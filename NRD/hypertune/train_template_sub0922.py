""" Tune the hyper-parameters. """
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='setsum_nn')
parser.add_argument('--code_embed_dim', type=int, default=100)
parser.add_argument('--fc_width', type=int, default=64)
parser.add_argument('--md_width', type=int, default=128, help='masked dense layer width')
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
parser.add_argument('--penalty_metric', type=str, default='cosine')
parser.add_argument('--count_cap', type=int, default=100)
parser.add_argument('--dx_rarecutpoint', type=int, default=10)
parser.add_argument('--pr_rarecutpoint', type=int, default=10)

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
tst_seed = args.tst_seed
n_fold = args.val_fold
result_file = args.result_file
penalty = args.penalty
penalty_metric = args.penalty_metric
count_cap = args.count_cap
DX_rarecutpoint = args.dx_rarecutpoint
PR_rarecutpoint = args.pr_rarecutpoint

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
from setsum_layer import SetSum, MaskedSum, MaskedDense, MaskedPooling
from glove import Glove
    
all_df = pd.read_csv(path+'cohorts/{}/{}_pred.csv'.format(cohort, cohort), dtype=core_dtypes_pd)
preprocessed = preprocess(all_df, DX_rarecutpoint=10, PR_rarecutpoint=10)
all_df = preprocessed['int_df']

tst_key = pd.read_csv(path+'cohorts/{}/tst_key{}.csv'.format(cohort, tst_seed), names = ['KEY_NRD'])
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

n_DX = 29
n_PR = 15
DXs = ['DX'+str(j) for j in range(2, n_DX+2)]
PRs = ['PR'+str(j) for j in range(1, n_PR+1)]

g = Glove(input_dim=len(code_cat), embedding_dim=code_embed_dim, count_cap=count_cap)
g.update_cooccur(train_df0[['DX1']+DXs+PRs])
g.train_glove(cache_path=model_path+'temp/{}/'.format(job_index), epochs=50, earlystop_patience=10, reducelr_patience=2, 
              parent_pairs=preprocessed['parent_pairs'], lamb=penalty, metric=penalty_metric, verbose=2)
embed_mat = g.get_embed_mat()

age_mean = train_df0['AGE'].mean()
age_std = train_df0['AGE'].std()
los_mean = train_df0['LOS'].mean()
los_std = train_df0['LOS'].std()

DX1_array_tst = tst_df['DX1'].values
DX_mat_tst = tst_df[DXs].values
PR_mat_tst = tst_df[PRs].values
hosp_array_tst = tst_df['HOSP_NRD'].values
    
demo_mat_tst = tst_df[['AGE', 'FEMALE']].values
demo_mat_tst[:, 0] = (demo_mat_tst[:, 0]-age_mean)/age_std
pay1_mat_tst = to_categorical(tst_df.PAY1.values)[:, 1:]
los_array_tst = (tst_df.LOS.values - los_mean)/los_std
ed_mat_tst = to_categorical(tst_df.HCUP_ED.values)
zipinc_mat_tst = to_categorical(tst_df.ZIPINC_QRTL.values)[:, 1:]
transfer_mat_tst = to_categorical(tst_df.SAMEDAYEVENT.values)
other_mat_tst = np.concatenate((demo_mat_tst, pay1_mat_tst, los_array_tst.reshape(los_array_tst.shape+(1,)), 
                                ed_mat_tst, zipinc_mat_tst, transfer_mat_tst), axis=1)
y_true = tst_df.readm30.astype(int).values

if embed_file=='random':
    embed_initializer = 'uniform'
else:
    embed_initializer = Constant(embed_mat)

#split trn/val data, do a n_fold validation
y_pred_lst = []
auc_lst = []
auc_freeze_lst = []
skf = StratifiedKFold(n_splits=n_fold, random_state=24, shuffle=True)
for trn_idx, val_idx in skf.split(train_df0, train_df0.HOSP_NRD):
    train_df = train_df0.copy()
    DX1_array_train = train_df['DX1'].values
    DX1_array_trn = DX1_array_train[trn_idx]
    DX1_array_val = DX1_array_train[val_idx]

    DX_mat_train = train_df[DXs].values
    DX_mat_trn = DX_mat_train[trn_idx, :]
    DX_mat_val = DX_mat_train[val_idx, :]

    PR_mat_train = train_df[PRs].values
    PR_mat_trn = PR_mat_train[trn_idx, :]
    PR_mat_val = PR_mat_train[val_idx, :]
    
    hosp_array_train = train_df['HOSP_NRD'].values
    hosp_array_trn = hosp_array_train[trn_idx]
    hosp_array_val = hosp_array_train[val_idx]
    
    demo_mat_train = train_df[['AGE', 'FEMALE']].values
    demo_mat_train[:, 0] = (demo_mat_train[:, 0]-age_mean)/age_std
    pay1_mat_train = to_categorical(train_df.PAY1.values)[:, 1:]
    los_array_train = (train_df.LOS.values - los_mean)/los_std
    ed_mat_train = to_categorical(train_df.HCUP_ED.values)
    zipinc_mat_train = to_categorical(train_df.ZIPINC_QRTL.values)[:, 1:]
    transfer_mat_train = to_categorical(train_df.SAMEDAYEVENT.values)
    other_mat_train = np.concatenate((demo_mat_train, pay1_mat_train, los_array_train.reshape(los_array_train.shape+(1,)), 
                                    ed_mat_train, zipinc_mat_train, transfer_mat_train), axis=1)
    other_mat_trn = other_mat_train[trn_idx, :]
    other_mat_val = other_mat_train[val_idx, :]

    y_train = train_df.readm30.astype(int).values
    Y_trn = to_categorical(y_train[trn_idx])
    Y_val = to_categorical(y_train[val_idx])
    
    # model building 
    input_DX1 = Input(shape=(1,))
    DX1_embed = Embedding(input_dim=len(code_cat), output_dim=code_embed_dim, embeddings_initializer=embed_initializer, 
                          name='DX1_embed')(input_DX1)
    DX1_embed = Reshape((code_embed_dim,))(DX1_embed)
    input_DX = Input(shape = (len(DXs),))
    DX_embed = Embedding(input_dim=len(code_cat), output_dim=code_embed_dim, mask_zero=True, embeddings_initializer=embed_initializer, 
                         name='DX_embed')(input_DX)
    if model_name=='setsum_nn' or 'setsum_lr':
        DX_embed = MaskedDense(md_width, activation='relu')(DX_embed)
        DX_embed = MaskedSum()(DX_embed)
    elif model_name=='embed_sum':
        DX_embed = MaskedSum()(DX_embed)
    elif model_name=='embed_pool':
        DX_embed = MaskedPooling()(DX_embed)
    input_PR = Input(shape = (len(PRs),))
    PR_embed = Embedding(input_dim=len(code_cat), output_dim=code_embed_dim, mask_zero=True, embeddings_initializer=embed_initializer, 
                         name='PR_embed')(input_PR)
    if model_name=='setsum_nn' or 'setsum_lr':
        PR_embed = MaskedDense(md_width, activation='relu')(PR_embed)
        PR_embed = MaskedSum()(PR_embed)
    elif model_name=='embed_sum':
        PR_embed = MaskedSum()(PR_embed)
    elif model_name=='embed_pool':
        PR_embed = MaskedPooling()(PR_embed)
    input_hosp = Input(shape=(1,))
    hosp_embed = Embedding(input_dim=len(hosp_cat), output_dim=hosp_embed_dim, input_length=1)(input_hosp)
    hosp_embed = Reshape((hosp_embed_dim, ))(hosp_embed)
    input_other = Input(shape=(other_mat_train.shape[1], ))
    merged = Concatenate(axis=1)([DX1_embed, DX_embed, PR_embed, hosp_embed, input_other])
    if model_name=='setsum_nn':
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
    
    class_weight = {0:(Y_trn.shape[0]/sum(Y_trn[:, 0])), 1:(Y_trn.shape[0]/sum(Y_trn[:, 1]))}
    
    hist = model.fit([DX1_array_trn, DX_mat_trn, PR_mat_trn, hosp_array_trn, other_mat_trn], Y_trn, 
                     batch_size=batchsize, epochs=40, callbacks=[auccheckpoint, reduce_lr, earlystop], class_weight=class_weight, 
                     validation_data=[[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val], Y_val], 
                    verbose=2)
    
    model.load_weights(model_path+'embeding_nn_sub_temp1_'+str(job_index)+'.h5')
    y = model.predict([DX1_array_tst, DX_mat_tst, PR_mat_tst, hosp_array_tst, other_mat_tst], verbose=0)
    y_pred = y[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    auc_freeze_lst.append(roc_auc)
    
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
    
    y = model.predict([DX1_array_tst, DX_mat_tst, PR_mat_tst, hosp_array_tst, other_mat_tst], verbose=0)
    y_pred = y[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    auc_lst.append(roc_auc)
    y_pred_lst.append(y_pred)

auc_mean = np.mean(auc_lst)
auc_freeze_mean = np.mean(auc_freeze_lst)
y_pred_mat = np.column_stack(y_pred_lst)
now = datetime.now().strftime('%y_%m_%d_%I_%M_%S')
y_pred_file = 'output/y_pred_mat'+now+'.npy'
np.save(y_pred_file, y_pred_mat)
y_pred_avg = y_pred_mat.mean(axis=1)
fpr, tpr, _ = roc_curve(y_true, y_pred_avg)
auc_avg = auc(fpr, tpr)
with open(result_file.format(job_index), 'a') as f:
    f.write('{0},{1},{2},{3},{4},{5:.1E},{6:.1E},{7:.1f},{8},{9},{10},{11},{12},{13:.1E},{14},{15},{16},{17},{18:.5f},{19:.5f},{20:.5f},{21}\n'.format(model_name, code_embed_dim, hosp_embed_dim, fc_width, md_width, lr1, lr2, dropout, batchsize, embed_file, cohort, tst_seed, n_fold, penalty, penalty_metric, count_cap, DX_rarecutpoint, PR_rarecutpoint, auc_mean, auc_avg, auc_freeze_mean, y_pred_file))
