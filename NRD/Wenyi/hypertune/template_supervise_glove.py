""" Tune the hyper-parameters. """
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='setsum_nn')
parser.add_argument('--code_embed_dim', type=int, default=100)
parser.add_argument('--fc_width', type=int, default=64)
parser.add_argument('--md_width', type=int, default=128, help='masked dense layer width')
parser.add_argument('--lr', type=float, default=0.0002)
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
parser.add_argument('--batchsize_ratio', type=int, default=12)
parser.add_argument('--loss_weight', type=float, default=1.)
parser.add_argument('--semi_proportion', type=float, default=1., help='proportion of training data with labels')

parser.add_argument('--job_index', type=int, default=0)

args = parser.parse_args()
model_name = args.model_name
code_embed_dim = args.code_embed_dim
fc_width = args.fc_width
md_width = args.md_width
lr = args.lr
dropout = args.dropout
readm_batchsize = args.batchsize
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
batchsize_ratio = args.batchsize_ratio
loss_weight = args.loss_weight
semi_proportion = args.semi_proportion

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
    
from keras.layers import Input, Embedding, Concatenate, Reshape, BatchNormalization, Multiply, Lambda, Add
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical, Sequence
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
from doublebatch import DoubleBatchGenerator

# Define useful and functions 
scaling_factor = 0.75
def weighting_factor(count):
    ge_mask = K.cast(K.greater_equal(count, count_cap), K.floatx())
    l_mask = K.cast(K.less(count, count_cap), K.floatx())
    return 1.*ge_mask + np.power(count/count_cap, scaling_factor)*l_mask
def loss_cooccur(y_true, y_pred):
    weight_cooccur = weighting_factor(K.exp(y_true))
    return K.mean(K.square(y_pred - y_true)*weight_cooccur)

#data reading and preprocessing
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
n_code = len(code_cat)-1

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

train_df = train_df0.sample(frac=semi_proportion).reset_index(drop=True)
n_train_sample = len(train_df)
auc_lst = []
kf2 = KFold(n_splits=n_fold, random_state=24, shuffle=True)
trn_idx, val_idx = next(kf2.split(train_df))

tst_gen = DoubleBatchGenerator(cooccur_df=cooccur_df, readm_df=tst_df, readm_batchsize=len(tst_df), 
                               batchsize_ratio=batchsize_ratio, shuffle=False)
parent_pairs = None
if True:
    trn_df = train_df.loc[trn_idx].reset_index(drop=True)
    val_df = train_df.loc[val_idx].reset_index(drop=True)
    trn_gen = DoubleBatchGenerator(cooccur_df=cooccur_df, readm_df=trn_df, readm_batchsize=readm_batchsize, 
                               batchsize_ratio=batchsize_ratio, shuffle=True)
    val_gen = DoubleBatchGenerator(cooccur_df=cooccur_df, readm_df=val_df, readm_batchsize=len(val_df), 
                               batchsize_ratio=batchsize_ratio, shuffle=False)
    val_x, val_y = next(iter(val_gen))
    
    #model definition
    input_w = Input(shape=(batchsize_ratio,), name='focal_index')
    input_v = Input(shape=(batchsize_ratio,), name='context_index')
    if parent_pairs is None:
        embed_layer = Embedding(input_dim=len(code_cat), output_dim=code_embed_dim, name='embed', 
                                embeddings_initializer=embed_initializer)
    else:
        embed_layer = Embedding(input_dim=len(code_cat), output_dim=code_embed_dim, name='embed',
                                embeddings_regularizer=Parent_reg(parent_pairs, lamb, metric))
    w_embed = embed_layer(input_w)
    v_embed = embed_layer(input_v)
    bias_layer = Embedding(input_dim=len(code_cat), output_dim=1, name='bias')
    w_bias = bias_layer(input_w)
    v_bias = bias_layer(input_v)
    
    inner = Multiply()([w_embed, v_embed])
    inner = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(inner)
    merged = Concatenate(axis=-1)([inner, w_bias, v_bias])
    output_cooccur = Lambda(lambda x: K.sum(x, axis=-1, keepdims=False), name='cooccur')(merged)
    
    input_DX = Input(shape = (len(DXs),))
    DX_embed = embed_layer(input_DX)
    DX_embed = MaskedDense(md_width, activation='relu')(DX_embed)
    DX_embed = MaskedSum()(DX_embed)
    input_PR = Input(shape = (len(PRs),))
    PR_embed = embed_layer(input_PR)
    PR_embed = MaskedDense(md_width, activation='relu')(PR_embed)
    PR_embed = MaskedSum()(PR_embed)
    input_other = Input(shape=(8, ))
    merged = Concatenate(axis=1)([DX_embed, PR_embed, input_other])
    merged = Dense(fc_width, activation='relu')(merged)
    merged = Dropout(dropout)(merged)
    output_readm = Dense(2, activation='softmax', name='readm')(merged)
    
    model = Model(inputs=[input_DX, input_PR, input_other, input_w, input_v], 
                  outputs=[output_readm, output_cooccur])
    
    adam = Adam(lr=lr)
    model.compile(loss={'readm':'categorical_crossentropy', 'cooccur':loss_cooccur}, 
                 optimizer=adam, loss_weights={'readm':1., 'cooccur':loss_weight})

    auccheckpoint = AUCCheckPoint(filepath=model_path+'embeding_nn_temp'+str(job_index)+'.h5', validation_x=val_x, validation_y=val_y,
                                  auc_output_idx=[0])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=K.epsilon())
    
    class_weight = {'readm':{0:1., 1:minor_class_weight}}
    
    hist = model.fit_generator(generator=trn_gen, validation_data=val_gen, epochs=80, verbose=2, 
                               callbacks=[auccheckpoint, reduce_lr], class_weight=class_weight)
    
    model.load_weights(model_path+'embeding_nn_temp'+str(job_index)+'.h5')
    y = model.predict_generator(tst_gen, verbose=1)
    y_pred = y[0][:, 1]
    y_true = tst_df.MORTALITY_1year.astype(int).values
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    auc_lst.append(roc_auc)
    #y_pred_lst.append(y_pred)
    
auc_mean = np.mean(auc_lst)
#y_pred_mat = np.column_stack(y_pred_lst)
#now = datetime.now().strftime('%y_%m_%d_%I_%M_%S')
#y_pred_file = path+'y_pred_mat/y_pred_mat'+now+'.npy'
#np.save(y_pred_file, y_pred_mat)
#y_pred_avg = y_pred_mat.mean(axis=1)
#fpr, tpr, _ = roc_curve(y_true, y_pred_avg)
#auc_avg = auc(fpr, tpr)
with open(result_file.format(job_index), 'a') as f:
    f.write('{},{},{},{},{:.1E},{:.1f},{},{},{},{},{},{},{:.1f},{},{:.1f},{:.1f},{:.5f},{},{},{}\n'.format(model_name, code_embed_dim, fc_width, md_width, lr, dropout, readm_batchsize, embed_file, tst_seed, n_fold, count_cap, code_rarecutpoint, minor_class_weight, batchsize_ratio, loss_weight, semi_proportion, auc_mean, n_sample, n_code, n_train_sample))
