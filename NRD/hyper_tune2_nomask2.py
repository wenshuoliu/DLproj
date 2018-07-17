""" Tune the hyper-parameters. """
import pandas as pd
import numpy as np
import os, sys
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from keras.layers import Input, Embedding, concatenate, Reshape, BatchNormalization, add, LSTM, CuDNNLSTM, Lambda
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
import keras.backend as K

module_path = '/home/wsliu/Codes/DLproj'
if module_path not in sys.path:
    sys.path.append(module_path)
from DL_utils import plot_roc
from keras_addon import AUCCheckPoint
from utils import Mat_reg

path = '/nfs/turbo/umms-awaljee/wsliu/Data/NRD/'
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
train_df = pd.concat([trn_df, val_df])
all_df = pd.concat([train_df, tst_df])

N_DX = 29
DXs = ['DX'+str(n) for n in range(2, N_DX+2)]
DX_series = pd.concat([all_df[DX] for DX in DXs])
DX_series = DX_series.fillna('missing')
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


DX_df = train_df[DXs]
DX_df = DX_df.fillna('missing')
for dx in DXs:
    DX_df[dx] = DX_df[dx].map(DX_dict)
DX_mat = DX_df.values
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
for dx in DXs:
    DX_df_tst[dx] = DX_df_tst[dx].map(DX_dict)
DX_mat_tst = DX_df_tst.values

demo_mat_tst = tst_df[['AGE', 'FEMALE']].values
demo_mat_tst[:, 0] = (demo_mat_tst[:, 0]-age_mean)/age_std

hosp_array_tst = tst_df['HOSP_NRD'].map(hosp_dict).values

DX1_series_tst = tst_df.DX1.map(DX1_dict)
DX1_mat_tst = np.zeros((len(tst_df), len(DX1_dict)))
for i, dx1 in enumerate(DX1_series_tst.values):
    DX1_mat_tst[i, dx1] = 1
    
y_tst = tst_df.readm30.astype(int).values

n_sample = 500

fc_widths = np.random.choice([16, 32, 64], n_sample)
DX_dims = np.random.randint(5, 50, n_sample)
hosp_dims = np.random.randint(1, 5, n_sample)
lrs = np.random.choice([0.0005, 0.0002, 0.0001], n_sample)
dropouts = np.random.rand(n_sample)*0.5
batch_sizes = np.random.choice([256, 128], n_sample)
penalty = np.power(10, np.random.rand(n_sample)*6-7)

parent_mat = np.load(path+'cohorts/ami/parent_mat2.npy')

parameters = list(zip(DX_dims, hosp_dims, penalty, fc_widths, dropouts, lrs, batch_sizes))

def model_build(DX_embed_dim,hosp_embed_dim,penalty,fc_width,dropout,lr,batch_size):
    input_DX = Input(shape = (N_DX,))
    DX_embed = Embedding(input_dim=parent_mat.shape[1], output_dim=DX_embed_dim, input_length=N_DX,
                     embeddings_regularizer=Mat_reg(parent_mat, penalty), name='DX_embed')(input_DX)
    DX_feature = CuDNNLSTM(DX_embed_dim, return_sequences=False)(DX_embed)
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
    return model

class_weight = {0:(Y_trn.shape[0]/sum(Y_trn[:, 0])), 1:(Y_trn.shape[0]/sum(Y_trn[:, 1]))}

for param in parameters:
    print("Training with parameters: ", *param)
    batch_size = param[-1]
    model = model_build(*param)
    auccheckpoint = AUCCheckPoint(filepath=model_path+'amiccs_lstm_nomask2_{}_{}_{:.4f}_{}_{:.4f}_{}_{}.h5'.format(*param), validation_y=Y_val[:, 1], validation_x=[demo_mat_val, DX1_mat_val, DX_mat_val, hosp_array_val])
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=K.epsilon())
    earlystop = EarlyStopping(monitor='val_loss', patience=50)
    hist = model.fit([demo_mat_trn, DX1_mat_trn, DX_mat_trn, hosp_array_trn], Y_trn, 
                 batch_size=batch_size, epochs=300, callbacks=[auccheckpoint, reduce_lr, earlystop], class_weight=class_weight, 
                 validation_data=[[demo_mat_val, DX1_mat_val, DX_mat_val, hosp_array_val], Y_val], 
                verbose=2)
        
    model.load_weights(model_path+'amiccs_lstm_nomask2_{}_{}_{:.4f}_{}_{:.4f}_{}_{}.h5'.format(*param))
    y = model.predict([demo_mat_tst, DX1_mat_tst, DX_mat_tst, hosp_array_tst], verbose=0)
    y_pred = y[:, 1]
    
    fpr, tpr, _ = roc_curve(y_tst, y_pred)
    roc_auc = auc(fpr, tpr)
    with open('output/hyper_tune_nomask_no18.csv', 'a') as f:
            f.write('{0},{1},{2},{3},{4},{5},{6},{7:.4f}\n'.format(*param,roc_auc))
    del(model)

