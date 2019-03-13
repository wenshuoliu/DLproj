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
parser.add_argument('--tst_seed', type=int, default=0, help='the seed to split training/test data')
parser.add_argument('--val_fold', type=int, default=10, help='number of folds to split training/validation data')
parser.add_argument('--result_file', type=str, default='output/result.csv')
#parser.add_argument('--penalty', type=float, default=0.5)
#parser.add_argument('--penalty_metric', type=str, default='cosine')
parser.add_argument('--count_cap', type=int, default=100)
parser.add_argument('--code_rarecutpoint', type=int, default=10)
#parser.add_argument('--clean_df', type=int, default=0, help='whether remove the patients with rare codes')
parser.add_argument('--class_weight', type=float, default=1.)

parser.add_argument('--job_index', type=int, default=0)

args = parser.parse_args()
model_name = args.model_name
code_embed_dim = args.code_embed_dim
fc_width = args.fc_width
md_width = args.md_width
lr1 = args.lr1
lr2 = args.lr2
dropout = args.dropout
batchsize = args.batchsize
embed_file = args.embed_file
tst_seed = args.tst_seed
n_fold = args.val_fold
result_file = args.result_file
#penalty = args.penalty
#penalty_metric = args.penalty_metric
count_cap = args.count_cap
code_rarecutpoint = args.code_rarecutpoint
#clean_df = args.clean_df
minor_class_weight = args.class_weight

job_index = args.job_index


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import roc_curve, auc
import os, sys, time
from datetime import datetime
import statsmodels.stats.api as sms

path = '/nfs/turbo/umms-awaljee/wsliu/Data/MIMIC/'
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
from keras.layers import Input, Embedding, Concatenate, Reshape, BatchNormalization, LSTM, CuDNNLSTM, CuDNNGRU, Lambda, Add
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
from glove import Glove
from setsum_layer import SetSum, MaskedSum, MaskedDense, MaskedPooling

n_DX = 30
n_PR = 15
DXs = ['DX'+str(n) for n in range(1, n_DX+1)]
PRs = ['PR'+str(n) for n in range(1, n_PR+1)]
all_df = pd.read_csv(path+'MIMIC3_Updated_include.csv')
all_df = all_df.drop(columns=['PR'+str(j) for j in range(15, 40)])
all_df.columns = ['Unnamed: 0', 'HADM_ID', 'MORTALITY_1year', 'AGE', 'GENDER', 'LOS', 'Eth'] + DXs + PRs

DX_series = pd.Series()
for dx in DXs:
    DX_series = pd.concat([DX_series, all_df[dx]])
PR_series = pd.Series()
for pr in PRs:
    PR_series = pd.concat([PR_series, all_df[pr]])

DX_freq = DX_series.value_counts()
PR_freq = PR_series.value_counts()
DX_cat = sorted(DX_freq.loc[DX_freq>=code_rarecutpoint].index)
PR_cat = sorted(PR_freq.loc[PR_freq>=code_rarecutpoint].index)
DX_in = set(DX_cat)
PR_in = set(PR_cat)
code_cat = ['missing'] + DX_cat + PR_cat
DX_dict = dict(zip(DX_cat, range(1, len(DX_cat)+1)))
PR_dict = dict(zip(PR_cat, range(len(DX_cat)+1, len(DX_cat)+len(PR_cat)+1)))
DX_dict['missing'] = 0
PR_dict['missing'] = 0

clean_df = all_df.copy()
for dx in DXs:
    clean_df.loc[~clean_df[dx].isin(DX_in), dx] = 'missing'
for pr in PRs:
    clean_df.loc[~clean_df[pr].isin(PR_in), pr] = 'missing'
clean_df = clean_df.loc[(clean_df[DXs] != 'missing').sum(axis=1)>0]
clean_df = clean_df.loc[(clean_df[PRs] != 'missing').sum(axis=1)>0]
        
all_df = clean_df.reset_index(drop=True)
n_sample = len(all_df)

int_df = all_df.copy()
for dx in DXs:
    int_df[dx] = int_df[dx].map(DX_dict)    
for pr in PRs:
    int_df[pr] = int_df[pr].map(PR_dict)
all_df = int_df.reset_index(drop=True)

kf = KFold(n_splits=5, random_state=tst_seed, shuffle=True)
train_idx, tst_idx = next(kf.split(all_df))
train_df0 = all_df.loc[train_idx].reset_index(drop=True)
tst_df = all_df.loc[tst_idx].reset_index(drop=True)

# GloVe pretraining
g = Glove(input_dim=len(code_cat), embedding_dim=code_embed_dim, count_cap=count_cap)
g.update_cooccur(train_df0[DXs+PRs])
cooccur_df = g.get_cooccur_df()
g.train_glove(cooccur_df=cooccur_df, cache_path=model_path+'temp/{}/'.format(job_index), epochs=80, earlystop_patience=10, 
reducelr_patience=3, batch_size=1024, verbose=2)
embed_mat = g.get_embed_mat()
if embed_file=='random':
    embed_initializer = 'uniform'
else:
    embed_initializer = Constant(embed_mat)

# Data formatting
DX_mat_tst = tst_df[DXs].values
PR_mat_tst = tst_df[PRs].values    
continue_mat_tst = tst_df[['AGE', 'GENDER', 'LOS']].values
eth_array_tst = tst_df['Eth'].values
eth_mat_tst = to_categorical(eth_array_tst, num_classes=5)
other_mat_tst = np.concatenate((continue_mat_tst, eth_mat_tst), axis=1)
y_true = tst_df.MORTALITY_1year.values

train_df = train_df0.copy()
y_pred_lst = []
auc_lst = []
auc_freeze_lst = []
kf2 = KFold(n_splits=n_fold, random_state=24, shuffle=True)
trn_idx, val_idx = next(kf2.split(train_df))
if True:
    DX_mat_train = train_df[DXs].values
    DX_mat_trn = DX_mat_train[trn_idx, :]
    DX_mat_val = DX_mat_train[val_idx, :]
    PR_mat_train = train_df[PRs].values        
    PR_mat_trn = PR_mat_train[trn_idx, :]
    PR_mat_val = PR_mat_train[val_idx, :]
    continue_mat_train = train_df[['AGE', 'GENDER', 'LOS']].values
    eth_array_train = train_df['Eth'].values
    eth_mat_train = to_categorical(eth_array_train, num_classes=5)
    other_mat_train = np.concatenate((continue_mat_train, eth_mat_train), axis=1)
    other_mat_trn = other_mat_train[trn_idx, :]
    other_mat_val = other_mat_train[val_idx, :]
    y_train = train_df.MORTALITY_1year.values
    y_trn = y_train[trn_idx]
    y_val = y_train[val_idx]
    Y_trn = to_categorical(y_train[trn_idx])
    Y_val = to_categorical(y_train[val_idx])
    
    #model definition
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
    input_other = Input(shape=(other_mat_train.shape[1], ))
    merged = Concatenate(axis=1)([DX_embed, PR_embed, input_other])
    if model_name=='setsum_nn':
        merged = Dense(fc_width, activation='relu')(merged)
        merged = Dropout(dropout)(merged)
    prediction = Dense(2, activation='softmax')(merged)
    model = Model(inputs=[input_DX, input_PR, input_other], outputs=prediction)
    
    for l in model.layers:
        if l.name=='DX_embed' or l.name=='PR_embed':
            l.trainable = False
    adam = Adam(lr=lr1)
    model.compile(optimizer=adam, loss='categorical_crossentropy')

    auccheckpoint = AUCCheckPoint(filepath=model_path+'embeding_nn_sub_temp1_'+str(job_index)+'.h5', validation_y=Y_val[:, 1], 
                                 validation_x=[DX_mat_val, PR_mat_val, other_mat_val])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=K.epsilon())
    earlystop = EarlyStopping(monitor='val_loss', patience=30)
    
    class_weight = {0:1., 1:minor_class_weight}
    
    hist = model.fit([DX_mat_trn, PR_mat_trn, other_mat_trn], Y_trn, 
                     batch_size=batchsize, epochs=80, callbacks=[auccheckpoint, reduce_lr, earlystop], class_weight=class_weight, 
                     validation_data=[[DX_mat_val, PR_mat_val, other_mat_val], Y_val], 
                    verbose=2)
    
    model.load_weights(model_path+'embeding_nn_sub_temp1_'+str(job_index)+'.h5')
    y = model.predict([DX_mat_tst, PR_mat_tst, other_mat_tst], verbose=0)
    y_pred = y[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    auc_freeze_lst.append(roc_auc)
    
    for l in model.layers:
        if l.name=='DX_embed' or l.name=='PR_embed':
            l.trainable = True
    adam = Adam(lr=lr2)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    
    auccheckpoint = AUCCheckPoint(filepath=model_path+'embeding_nn_sub_temp2_'+str(job_index)+'.h5', validation_y=Y_val[:, 1], 
                                 validation_x=[DX_mat_val, PR_mat_val, other_mat_val])
    hist = model.fit([DX_mat_trn, PR_mat_trn, other_mat_trn], Y_trn, 
                     batch_size=batchsize, epochs=30, callbacks=[auccheckpoint, reduce_lr, earlystop], class_weight=class_weight, 
                     validation_data=[[DX_mat_val, PR_mat_val, other_mat_val], Y_val], 
                    verbose=2)
    
    y = model.predict([DX_mat_tst, PR_mat_tst, other_mat_tst], verbose=0)
    y_pred = y[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    auc_lst.append(roc_auc)
    y_pred_lst.append(y_pred)
    
auc_mean = np.mean(auc_lst)
auc_freeze_mean = np.mean(auc_freeze_lst)
#y_pred_mat = np.column_stack(y_pred_lst)
#now = datetime.now().strftime('%y_%m_%d_%I_%M_%S')
#y_pred_file = path+'y_pred_mat/y_pred_mat'+now+'.npy'
#np.save(y_pred_file, y_pred_mat)
#y_pred_avg = y_pred_mat.mean(axis=1)
#fpr, tpr, _ = roc_curve(y_true, y_pred_avg)
#auc_avg = auc(fpr, tpr)
with open(result_file.format(job_index), 'a') as f:
    f.write('{},{},{},{},{:.1E},{:.1E},{:.1f},{},{},{},{},{},{},{:.1f},{:.5f},{:.5f},{}\n'.format(model_name, code_embed_dim, fc_width, md_width, lr1, lr2, dropout, batchsize, embed_file, tst_seed, n_fold, count_cap, code_rarecutpoint, minor_class_weight, auc_mean, auc_freeze_mean, n_sample))
