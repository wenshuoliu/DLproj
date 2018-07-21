""" Tune the hyper-parameters. """
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--sum_layer', type=int, default=0, help='1 is SetSum, 0 is MaskedSum')
parser.add_argument('--rare_cutpoint', type=int, default=1)
parser.add_argument('--dx_dim', type=int, default=50)
parser.add_argument('--hosp_dim', type=int, default=1)
parser.add_argument('--penalty', type=float, default=0.001)
parser.add_argument('--fc_width', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batchsize', type=int, default=128)

args = parser.parse_args()
rare_cutpoint = args.rare_cutpoint
sum_layer = args.sum_layer
DX_embed_dim = args.dx_dim
hosp_embed_dim = args.hosp_dim
penalty = args.penalty
fc_width = args.fc_width
lr = args.lr
dropout = args.dropout
batchsize = args.batchsize

import pandas as pd
import numpy as np
import os, sys
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from keras.layers import Input, Embedding, concatenate, LSTM, CuDNNLSTM, Lambda, Reshape
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
import keras.backend as K

module_path = '/home/wsliu/Codes/DLproj'
if module_path not in sys.path:
    sys.path.append(module_path)
if module_path+'/NRD' not in sys.path:
    sys.path.append(module_path+'/NRD')
from DL_utils import plot_roc
from keras_addon import AUCCheckPoint
from utils import Mat_reg
from setsum_layer import SetSum, MaskedSum

#path = '/nfs/turbo/umms-awaljee/wsliu/Data/NRD/'
path = '/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/NRD/'
model_path = path + 'models/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)
    
core_dtypes_pd = {'AGE': float,
 'AWEEKEND': float,
 'DIED': float,
 'DISCWT': float,
 'DISPUNIFORM': float,
 'DMONTH': float,
 'DQTR': float,
 'DRG': float,
 'DRGVER': float,
 'DRG_NoPOA': float,
 'DX1': bytes,
 'DX10': bytes,
 'DX11': bytes,
 'DX12': bytes,
 'DX13': bytes,
 'DX14': bytes,
 'DX15': bytes,
 'DX16': bytes,
 'DX17': bytes,
 'DX18': bytes,
 'DX19': bytes,
 'DX2': bytes,
 'DX20': bytes,
 'DX21': bytes,
 'DX22': bytes,
 'DX23': bytes,
 'DX24': bytes,
 'DX25': bytes,
 'DX26': bytes,
 'DX27': bytes,
 'DX28': bytes,
 'DX29': bytes,
 'DX3': bytes,
 'DX30': bytes,
 'DX4': bytes,
 'DX5': bytes,
 'DX6': bytes,
 'DX7': bytes,
 'DX8': bytes,
 'DX9': bytes,
 'DXCCS1': float,
 'DXCCS10': float,
 'DXCCS11': float,
 'DXCCS12': float,
 'DXCCS13': float,
 'DXCCS14': float,
 'DXCCS15': float,
 'DXCCS16': float,
 'DXCCS17': float,
 'DXCCS18': float,
 'DXCCS19': float,
 'DXCCS2': float,
 'DXCCS20': float,
 'DXCCS21': float,
 'DXCCS22': float,
 'DXCCS23': float,
 'DXCCS24': float,
 'DXCCS25': float,
 'DXCCS26': float,
 'DXCCS27': float,
 'DXCCS28': float,
 'DXCCS29': float,
 'DXCCS3': float,
 'DXCCS30': float,
 'DXCCS4': float,
 'DXCCS5': float,
 'DXCCS6': float,
 'DXCCS7': float,
 'DXCCS8': float,
 'DXCCS9': float,
 'ECODE1': bytes,
 'ECODE2': bytes,
 'ECODE3': bytes,
 'ECODE4': bytes,
 'ELECTIVE': float,
 'E_CCS1': float,
 'E_CCS2': float,
 'E_CCS3': float,
 'E_CCS4': float,
 'FEMALE': float,
 'HCUP_ED': float,
 'HOSP_NRD': float,
 'KEY_NRD': float,
 'LOS': float,
 'MDC': float,
 'MDC_NoPOA': float,
 'NCHRONIC': float,
 'NDX': float,
 'NECODE': float,
 'NPR': float,
 'NRD_DaysToEvent': float,
 'NRD_STRATUM': float,
 'NRD_VisitLink': bytes,
 'ORPROC': float,
 'PAY1': float,
 'PL_NCHS': float,
 'PR1': bytes,
 'PR10': bytes,
 'PR11': bytes,
 'PR12': bytes,
 'PR13': bytes,
 'PR14': bytes,
 'PR15': bytes,
 'PR2': bytes,
 'PR3': bytes,
 'PR4': bytes,
 'PR5': bytes,
 'PR6': bytes,
 'PR7': bytes,
 'PR8': bytes,
 'PR9': bytes,
 'PRCCS1': float,
 'PRCCS10': float,
 'PRCCS11': float,
 'PRCCS12': float,
 'PRCCS13': float,
 'PRCCS14': float,
 'PRCCS15': float,
 'PRCCS2': float,
 'PRCCS3': float,
 'PRCCS4': float,
 'PRCCS5': float,
 'PRCCS6': float,
 'PRCCS7': float,
 'PRCCS8': float,
 'PRCCS9': float,
 'PRDAY1': float,
 'PRDAY10': float,
 'PRDAY11': float,
 'PRDAY12': float,
 'PRDAY13': float,
 'PRDAY14': float,
 'PRDAY15': float,
 'PRDAY2': float,
 'PRDAY3': float,
 'PRDAY4': float,
 'PRDAY5': float,
 'PRDAY6': float,
 'PRDAY7': float,
 'PRDAY8': float,
 'PRDAY9': float,
 'REHABTRANSFER': float,
 'RESIDENT': float,
 'SAMEDAYEVENT': bytes,
 'SERVICELINE': float,
 'TOTCHG': float,
 'YEAR': float,
 'ZIPINC_QRTL': float}

train_df = pd.read_csv(path+'cohorts/ami/DX_train.csv', dtype=core_dtypes_pd)
tst_df = pd.read_csv(path+'cohorts/ami/DX_tst.csv', dtype=core_dtypes_pd)

trn_df, val_df = train_test_split(train_df, test_size=0.11, stratify=train_df.HOSP_NRD)
N_trn = len(trn_df)
N_val = len(val_df)
all_df = pd.concat([train_df, tst_df])

print("Define the dictionaries of DX, DX1, hospital...")
N_DX = 29
DXs = ['DX'+str(n) for n in range(2, N_DX+2)]

DX_series = pd.concat([all_df[DX] for DX in DXs])
DX_series = DX_series.fillna('missing')
DX_series[DX_series.isin(['invl', 'incn'])] = 'missing'

multi_ccs = pd.read_csv(path+'ccs_multi_dx_tool_2015.csv')
multi_ccs.columns = ['ICD9CM_CODE', 'CCS_LVL1', 'CCS_LVL1_LABEL', 'CCS_LVL2', 'CCS_LVL2_LABEL', 'CCS_LVL3', 'CCS_LVL3_LABEL', 
                    'CCS_LVL4', 'CCS_LVL4_LABEL']
multi_ccs.ICD9CM_CODE = multi_ccs.ICD9CM_CODE.apply(lambda x:x.replace("'", "").replace(' ', ''))
for j in range(1, 5):
    multi_ccs['CCS_LVL'+str(j)] = multi_ccs['CCS_LVL'+str(j)].apply(lambda x:x.replace("'", "").replace(' ', ''))
multi_ccs = multi_ccs[['ICD9CM_CODE', 'CCS_LVL1', 'CCS_LVL2', 'CCS_LVL3', 'CCS_LVL4']]

unclassified = set(multi_ccs.loc[multi_ccs.CCS_LVL1 == '18', 'ICD9CM_CODE'])
DX_series.loc[DX_series.isin(unclassified)] = 'missing'

code_freq = DX_series.value_counts()
rare_code = code_freq[code_freq<rare_cutpoint].index
DX_series.loc[DX_series.isin(rare_code)] = 'missing'

DX_series = DX_series.astype('category')
DX_cat = DX_series.cat.categories
n_DX_cat = len(DX_cat)
DX_int_cat = list(range(n_DX_cat)[1:]) +[0]
DX_dict = dict(zip(DX_cat, DX_int_cat))

DX1_series = all_df['DX1'].astype('category')
DX1_cat = DX1_series.cat.categories
DX1_int_cat = range(len(DX1_cat))
DX1_dict = dict(zip(DX1_cat, DX1_int_cat))

hosp_series = all_df['HOSP_NRD'].astype('category')
hosp_cat = hosp_series.cat.categories
hosp_dict = dict(zip(hosp_cat, range(len(hosp_cat))))

print("Define the parent matrix...")
ami_ccs = multi_ccs.loc[multi_ccs.ICD9CM_CODE.isin(DX_cat)]
ccs_cat = pd.concat([ami_ccs.CCS_LVL1, ami_ccs.CCS_LVL2, ami_ccs.CCS_LVL3, ami_ccs.CCS_LVL4]).astype('category').cat.categories
all_codes = [DX_cat[-1]]+list(DX_cat[:-1]) + list(ccs_cat[1:])

ami_ccs = ami_ccs.set_index(ami_ccs.ICD9CM_CODE, drop=True).drop(['ICD9CM_CODE'], axis=1)
ami_ccs['finest_non_empty'] = ami_ccs.CCS_LVL4
ami_ccs.finest_non_empty[ami_ccs.finest_non_empty==''] = ami_ccs.CCS_LVL3[ami_ccs.finest_non_empty == '']
ami_ccs.finest_non_empty[ami_ccs.finest_non_empty==''] = ami_ccs.CCS_LVL2[ami_ccs.finest_non_empty == '']
ami_ccs.finest_non_empty[ami_ccs.finest_non_empty==''] = ami_ccs.CCS_LVL1[ami_ccs.finest_non_empty == '']
DX_parent = ami_ccs[['finest_non_empty']]
DX_parent.columns = ['parent_code']
DX_parent = DX_parent.assign(parent_index=DX_parent.parent_code.apply(lambda x:all_codes.index(x)).values)
DX_parent = DX_parent.append(pd.DataFrame(index=['missing'], data=dict(parent_code = ['NA'], 
                                                                                       parent_index=[-1])))

CCS_parent = pd.DataFrame(dict(CCS_code=ccs_cat[1:]))
CCS_parent = CCS_parent.assign(parent_code=CCS_parent.CCS_code.apply(lambda x:'.'.join(x.split('.')[:-1])))
def get_CCS_ind(code):
    try:
        ind = all_codes.index(code)
    except ValueError:
        ind = -1
    return ind
CCS_parent = CCS_parent.assign(parent_index=CCS_parent.parent_code.apply(get_CCS_ind).values)
CCS_parent = CCS_parent.set_index(CCS_parent.CCS_code, drop=True).drop(['CCS_code'], axis=1)
parent_df = pd.concat([DX_parent, CCS_parent], axis=0)

parent_mat = np.identity(len(all_codes))
for i, c in enumerate(all_codes):
    parent_ind = parent_df.loc[c, 'parent_index']
    if not parent_ind == -1:
        parent_mat[i, parent_ind] = -1
parent_mat = parent_mat[parent_mat.sum(axis=1)==0, :]

print("Data preparation...")
DX_df = train_df[DXs]
DX_df = DX_df.fillna('missing')
DX_df[DX_df.isin(['invl', 'incn'])] = 'missing'
DX_df[DX_df.isin(rare_code)] = 'missing'
DX_df[DX_df.isin(unclassified)] = 'missing'
for dx in DXs:
    DX_df[dx] = DX_df[dx].map(DX_dict)
DX_mat = DX_df.values
DX_mat.sort(axis=1)
DX_mat_trn = DX_df.values[:N_trn, ]
DX_mat_val = DX_df.values[N_trn:, ]

demo_mat = train_df[['AGE', 'FEMALE']].values
age_mean = train_df['AGE'].mean()
age_std = train_df['AGE'].std()
demo_mat[:, 0] = (demo_mat[:, 0]-age_mean)/age_std
demo_mat_trn = demo_mat[:N_trn, ]
demo_mat_val = demo_mat[N_trn:, ]

hosp_series = train_df['HOSP_NRD'].map(hosp_dict)
hosp_array = hosp_series.values
hosp_array_trn = hosp_array[:N_trn]
hosp_array_val = hosp_array[N_trn:]

DX1_series = train_df['DX1'].map(DX1_dict)
DX1_mat = np.zeros((len(DX1_series), len(DX1_dict)))
for i, dx1 in enumerate(DX1_series.values):
    DX1_mat[i, dx1] = 1
DX1_mat_trn = DX1_mat[:N_trn, ]
DX1_mat_val = DX1_mat[N_trn:, ]

y = train_df['readm30'].values.astype(int)
Y_trn = to_categorical(y[:N_trn])
Y_val = to_categorical(y[N_trn:])

DX_df_tst = tst_df[DXs]
DX_df_tst = DX_df_tst.fillna('missing')
DX_df_tst[DX_df_tst.isin(['invl', 'incn'])]
DX_df_tst[DX_df_tst.isin(rare_code)] = 'missing'
DX_df_tst[DX_df_tst.isin(unclassified)] = 'missing'
for dx in DXs:
    DX_df_tst[dx] = DX_df_tst[dx].map(DX_dict)
DX_mat_tst = DX_df_tst.values
DX_mat_tst.sort(axis=1)

demo_mat_tst = tst_df[['AGE', 'FEMALE']].values
demo_mat_tst[:, 0] = (demo_mat_tst[:, 0]-age_mean)/age_std

hosp_array_tst = tst_df['HOSP_NRD'].map(hosp_dict).values

DX1_series_tst = tst_df.DX1.map(DX1_dict)
DX1_mat_tst = np.zeros((len(tst_df), len(DX1_dict)))
for i, dx1 in enumerate(DX1_series_tst.values):
    DX1_mat_tst[i, dx1] = 1
    
y_tst = tst_df.readm30.astype(int).values

print("Model building and training...")
input_DX = Input(shape = (N_DX,))
DX_embed = Embedding(input_dim=parent_mat.shape[1], output_dim=DX_embed_dim, mask_zero=True,
                     embeddings_regularizer=Mat_reg(parent_mat, 0.01), name='DX_embed')(input_DX)
if sum_layer == 1:
    DX_feature = SetSum(DX_embed_dim, activation='tanh')(DX_embed)
else:
    DX_feature = MaskedSum()(DX_embed)    
input_demo = Input(shape=(2, ))
input_DX1 = Input(shape=(len(DX1_cat),))
input_hosp = Input(shape=(1,))
hosp_embed = Embedding(input_dim=len(hosp_cat), output_dim=hosp_embed_dim, input_length=1)(input_hosp)
hosp_embed = Reshape((hosp_embed_dim, ))(hosp_embed)
merged = concatenate([input_demo, input_DX1, DX_feature, hosp_embed], axis=1)
x = Dense(fc_width, activation='relu')(merged)
x = Dropout(dropout)(x)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=[input_demo, input_DX1, input_DX, input_hosp], outputs=prediction)
adam = Adam(lr=lr)
model.compile(optimizer=adam, loss='categorical_crossentropy')

class_weight = {0:(Y_trn.shape[0]/sum(Y_trn[:, 0])), 1:(Y_trn.shape[0]/sum(Y_trn[:, 1]))}

auccheckpoint = AUCCheckPoint(filepath=model_path+'amiccs_lstm_masksum_temp.h5', validation_y=Y_val[:, 1], validation_x=[demo_mat_val, DX1_mat_val, DX_mat_val, hosp_array_val])
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=K.epsilon())
earlystop = EarlyStopping(monitor='val_loss', patience=30)

hist = model.fit([demo_mat_trn, DX1_mat_trn, DX_mat_trn, hosp_array_trn], Y_trn, 
                 batch_size=batchsize, epochs=200, callbacks=[auccheckpoint, reduce_lr, earlystop], class_weight=class_weight, 
                 validation_data=[[demo_mat_val, DX1_mat_val, DX_mat_val, hosp_array_val], Y_val], 
                verbose=2)
        
model.load_weights(model_path+'amiccs_lstm_masksum_temp.h5')
y = model.predict([demo_mat_tst, DX1_mat_tst, DX_mat_tst, hosp_array_tst], verbose=0)
y_pred = y[:, 1]
    
fpr, tpr, _ = roc_curve(y_tst, y_pred)
roc_auc = auc(fpr, tpr)
with open('output/hypertune_masksum0.csv', 'a') as f:
            f.write('{0},{1},{2},{3},{4:.4f},{5},{6:.4f},{7:.4f},{8},{9:.4f}\n'.format(sum_layer, rare_cutpoint, DX_embed_dim, hosp_embed_dim, penalty, fc_width, lr, dropout, batchsize,roc_auc))