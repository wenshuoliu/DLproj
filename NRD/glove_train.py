import pandas as pd
import numpy as np
import os
from glove import Glove
from ccs_tools import dx_multi, pr_multi

DX_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE)
PR_cat = ['missing'] + sorted(pr_multi.ICD9CM_CODE)
code_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE) + sorted(pr_multi.ICD9CM_CODE)

path = '/nfs/turbo/umms-awaljee/wsliu/Data/NRD/'
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
g = Glove(input_dim=len(code_cat), embedding_dim=100)
cooccur_df = pd.read_csv(path+'all/cooccur_df.csv')
g.train_glove(cooccur_df=cooccur_df, cache_path=model_path, epochs=100, verbose=2)

embed_mat = g.get_embed_mat()

np.save(path+'all/embed_mat0816.npy', embed_mat)