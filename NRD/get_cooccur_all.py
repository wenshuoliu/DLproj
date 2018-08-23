# Get the cooccurrences of DX and PR codes from the whole NRD database
import pandas as pd
import numpy as np
import os, time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from glove import Glove
from ccs_tools import dx_multi, pr_multi
from utils import core_dtypes_pd, core_cols, na_values

path = '/nfs/turbo/umms-awaljee/wsliu/Data/NRD/'
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
DX1_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE)
DX_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE)
PR_cat = ['missing'] + sorted(pr_multi.ICD9CM_CODE)
code_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE) + sorted(dx_multi.ICD9CM_CODE) + sorted(pr_multi.ICD9CM_CODE)

DX1_dict = dict(zip(DX1_cat, range(len(DX_cat))))
DX_dict = dict(zip(DX_cat, [0] + list(range(len(DX_cat), len(DX_cat)*2))))
PR_dict = dict(zip(PR_cat, [0] + list(range(len(DX_cat)*2-1, len(DX_cat)*2+len(PR_cat)-1))))

DXs = ['DX'+str(j) for j in range(2, 31)]
PRs = ['PR'+str(j) for j in range(1, 16)]

unclassified = set(dx_multi.loc[dx_multi.CCS_LVL1 == '18', 'ICD9CM_CODE'])

g = Glove(input_dim=len(code_cat), embedding_dim=100)

#dtypes = dict(zip(DXs, [bytes]*30))
#dtypes.update(zip(PRs, [bytes]*15))

dxpr_df = pd.read_csv(path+'raw/2014/NRD_2014_Core.CSV', sep=',', 
                           header = None, 
                           names=core_cols, 
                           dtype=core_dtypes_pd, 
                           na_values=na_values, 
                           chunksize=500000)

chunk_id = 0
for df in dxpr_df:
    start = time.time()
    DX1_df = df[['DX1']]
    DX1_df = DX1_df.fillna('missing')
    DX1_df[DX1_df.isin(['invl', 'incn'])] = 'missing'
    DX1_df[DX1_df.isin(unclassified)] = 'missing'
    DX1_df['DX1'] = DX1_df['DX1'].map(DX1_dict)
        
    DX_df = df[DXs]
    DX_df = DX_df.fillna('missing')
    DX_df[DX_df.isin(['invl', 'incn'])] = 'missing'
    DX_df[DX_df.isin(unclassified)] = 'missing'
    for dx in DXs:
        DX_df[dx] = DX_df[dx].map(DX_dict)
        
    PR_df = df[PRs]
    PR_df = PR_df.fillna('missing')
    PR_df[PR_df.isin(['invl', 'incn'])] = 'missing'
    for pr in PRs:
        PR_df[pr] = PR_df[pr].map(PR_dict)
        
    df = pd.concat([DX1_df, DX_df, PR_df], axis=1)
    g.update_cooccur(df)
    print('Chunk {} finished. It takes {:.1f} seconds.'.format(chunk_id, time.time()-start))
    chunk_id += 1
    
cooccur_df = g.get_cooccur_df()
cooccur_df.to_csv(path+'all/cooccur_df0820.csv', index=False)
