import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import os, sys
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

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from DL_utils import plot_roc
from keras_addon import AUCCheckPoint
from utils import Mat_reg, core_dtypes_pd

from ccs_tools import dx_multi, pr_multi
from setsum_layer import SetSum, MaskedSum, MaskedDense

train_df = pd.read_csv(path+'cohorts/ami/DX_train.csv', dtype=core_dtypes_pd)
tst_df = pd.read_csv(path+'cohorts/ami/DX_tst.csv', dtype=core_dtypes_pd)
all_df = pd.concat([train_df, tst_df])

DX1_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE)
DX_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE)
PR_cat = ['missing'] + sorted(pr_multi.ICD9CM_CODE)
code_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE) + sorted(dx_multi.ICD9CM_CODE) + sorted(pr_multi.ICD9CM_CODE)
n_DX_cat = len(DX_cat)
n_PR_cat = len(PR_cat)
n_code_cat = len(code_cat)
DX1_dict = dict(zip(DX1_cat, range(len(DX_cat))))
DX_dict = dict(zip(DX_cat, [0] + list(range(len(DX_cat), len(DX_cat)*2))))
PR_dict = dict(zip(PR_cat, [0] + list(range(len(DX_cat)*2-1, len(DX_cat)*2+len(PR_cat)-1))))
n_DX = 29
n_PR = 15
DXs = ['DX'+str(j) for j in range(2, n_DX+2)]
PRs = ['PR'+str(j) for j in range(1, n_PR+1)]

unclassified = set(dx_multi.loc[dx_multi.CCS_LVL1 == '18', 'ICD9CM_CODE'])

hosp_series = all_df['HOSP_NRD'].astype('category')
hosp_cat = hosp_series.cat.categories
hosp_dict = dict(zip(hosp_cat, range(len(hosp_cat))))

age_mean = train_df['AGE'].mean()
age_std = train_df['AGE'].std()
los_mean = train_df['LOS'].mean()
los_std = train_df['LOS'].std()

DX1_array_tst = tst_df.DX1.map(DX_dict).values
DX_df_tst = tst_df[DXs]
DX_df_tst = DX_df_tst.fillna('missing')
DX_df_tst[DX_df_tst.isin(['invl', 'incn'])]
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
transfer_mat_tst = to_categorical(tst_df.SAMEDAYEVENT.values)
other_mat_tst = np.concatenate((demo_mat_tst, pay1_mat_tst, los_array_tst.reshape(los_array_tst.shape+(1,)), 
                                ed_mat_tst, zipinc_mat_tst, transfer_mat_tst), axis=1)

n_fold = 10
code_embed_dim = 100
hosp_embed_dim = 1
    
embed_glove = np.load(path+'cohorts/ami/embed_mat_ami0824.npy')
#embed_glove = np.concatenate((embed_glove[:n_DX_cat, :], embed_glove[1:, :]), axis=0)
    
fc_width = 64
md_width = 128

for fold_ind in range(n_fold):
    trn_df, val_df = train_test_split(train_df, test_size=0.11, stratify=train_df.HOSP_NRD)
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
    transfer_mat = to_categorical(train_df.SAMEDAYEVENT.values)
    other_mat = np.concatenate((demo_mat, pay1_mat, los_array.reshape(los_array.shape+(1,)), ed_mat, zipinc_mat, 
                            transfer_mat), axis=1)
    other_mat_trn = other_mat[:N_trn, ]
    other_mat_val = other_mat[N_trn:, ]
    
    y = train_df['readm30'].values.astype(int)
    Y_trn = to_categorical(y[:N_trn])
    Y_val = to_categorical(y[N_trn:])
    
    input_DX1 = Input(shape=(1,))
    DX1_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, embeddings_initializer=Constant(embed_glove), 
                         name='DX1_embed')(input_DX1)
    DX1_embed = Reshape((code_embed_dim,))(DX1_embed)
    input_DX = Input(shape = (n_DX,))
    DX_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, mask_zero=True, embeddings_initializer=Constant(embed_glove), 
                         name='DX_embed')(input_DX)
    DX_feature = MaskedDense(md_width, activation='relu')(DX_embed)
    DX_feature = MaskedSum()(DX_feature)
    input_PR = Input(shape = (n_PR,))
    PR_embed = Embedding(input_dim=n_code_cat, output_dim=code_embed_dim, mask_zero=True, embeddings_initializer=Constant(embed_glove),
                         name='PR_embed')(input_PR)
    PR_feature = MaskedDense(md_width, activation='relu')(PR_embed)
    PR_feature = MaskedSum()(PR_feature)
    input_hosp = Input(shape=(1,))
    hosp_embed = Embedding(input_dim=len(hosp_cat), output_dim=hosp_embed_dim, input_length=1)(input_hosp)
    hosp_embed = Reshape((hosp_embed_dim, ))(hosp_embed)
    input_other = Input(shape=(other_mat.shape[1], ))
    merged = Concatenate(axis=1)([DX1_embed, DX_feature, PR_feature, hosp_embed, input_other])
    x = Dense(fc_width, activation='relu')(merged)
    x = Dropout(0.3)(x)
    prediction = Dense(2, activation='softmax')(x)
    model = Model(inputs=[input_DX1, input_DX, input_PR, input_hosp, input_other], outputs=prediction)
    
    for l in model.layers:
        if l.name=='DX_embed' or l.name=='PR_embed' or l.name=='DX1_embed':
            l.trainable = False
            
    adam = Adam(lr=0.0002)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    
    auccheckpoint = AUCCheckPoint(filepath=model_path+'ami_glove_auc_temp.h5', validation_y=Y_val[:, 1], 
                                 validation_x=[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=K.epsilon())
    earlystop = EarlyStopping(monitor='val_loss', patience=20)
    
    class_weight = {0:(Y_trn.shape[0]/sum(Y_trn[:, 0])), 1:(Y_trn.shape[0]/sum(Y_trn[:, 1]))}
    
    hist = model.fit([DX1_array_trn, DX_mat_trn, PR_mat_trn, hosp_array_trn, other_mat_trn], Y_trn, 
                     batch_size=128, epochs=50, callbacks=[auccheckpoint, reduce_lr, earlystop], class_weight=class_weight, 
                     validation_data=[[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val], Y_val], 
                    verbose=2)
    
    model.load_weights(model_path+'ami_glove_auc_temp.h5')

    for l in model.layers:
        if l.name=='DX_embed' or l.name=='PR_embed' or l.name=='DX1_embed':
            l.trainable = True
    adam = Adam(lr=0.00002)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    
    auccheckpoint = AUCCheckPoint(filepath=model_path+'ami_glove_auc_temp2.h5', validation_y=Y_val[:, 1], 
                                 validation_x=[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val])
    hist = model.fit([DX1_array_trn, DX_mat_trn, PR_mat_trn, hosp_array_trn, other_mat_trn], Y_trn, 
                     batch_size=128, epochs=10, callbacks=[auccheckpoint, reduce_lr, earlystop], class_weight=class_weight, 
                     validation_data=[[DX1_array_val, DX_mat_val, PR_mat_val, hosp_array_val, other_mat_val], Y_val], 
                    verbose=2)
    
    model.load_weights(model_path+'ami_glove_auc_temp2.h5')
    
    y = model.predict([DX1_array_tst, DX_mat_tst, PR_mat_tst, hosp_array_tst, other_mat_tst], verbose=0)
    y_pred = y[:, 1]
    np.save('output/y_pred'+str(fold_ind)+'.npy', y_pred)
    
    