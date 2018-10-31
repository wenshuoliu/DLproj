import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--embed_dim', type=int, default=100)
parser.add_argument('--penalty', type=float, default=1.)
parser.add_argument('--cooccur_file', type=str, default='all/cooccur_df.csv')
parser.add_argument('--test_fold', type=int, default=0, help='the random_state used to split train/tst')
parser.add_argument('--sep_dx1', type=int, default=0, help='whether separate DX1 when training embedding')
parser.add_argument('--count_cap', type=int, default=100, help='the count_cap of Glove')
parser.add_argument('--cohort', type=str, default='ami')
parser.add_argument('--save_folder', type=str, default='all/sepdx1/')
parser.add_argument('--metric', type=str, default='cosine')

parser.add_argument('--job_index', type=int, default=0)

args = parser.parse_args()
embed_dim = args.embed_dim
lamb = args.penalty
cooccur_file = args.cooccur_file
test_fold = args.test_fold
sep_dx1 = args.sep_dx1
count_cap = args.count_cap
cohort = args.cohort
save_folder = args.save_folder
metric = args.metric

job_index = args.job_index

import pandas as pd
import numpy as np
import os, sys

module_path = '/home/wsliu/Codes/DLproj'
if module_path not in sys.path:
    sys.path.append(module_path)
if module_path+'/NRD' not in sys.path:
    sys.path.append(module_path+'/NRD')
    
from glove import Glove
from ccs_tools import dx_multi, pr_multi

path = '/nfs/turbo/umms-awaljee/wsliu/Data/NRD/'
model_path = path + 'models/temp/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)

DX_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE)
PR_cat = ['missing'] + sorted(pr_multi.ICD9CM_CODE)
n_DX_cat = len(DX_cat)
n_PR_cat = len(PR_cat)
unclassified = set(dx_multi.loc[dx_multi.CCS_LVL1 == '18', 'ICD9CM_CODE'])
dx_ccs_cat = pd.concat([dx_multi.CCS_LVL1, dx_multi.CCS_LVL2, dx_multi.CCS_LVL3, dx_multi.CCS_LVL4]).astype('category').cat.categories
pr_ccs_cat = pd.concat([pr_multi.CCS_LVL1, pr_multi.CCS_LVL2, pr_multi.CCS_LVL3]).astype('category').cat.categories

if sep_dx1==1:
    DX1_dict = dict(zip(DX_cat, range(len(DX_cat))))
    DX_dict = dict(zip(DX_cat, [0] + list(range(len(DX_cat), len(DX_cat)*2))))
    PR_dict = dict(zip(PR_cat, [0] + list(range(len(DX_cat)*2-1, len(DX_cat)*2+len(PR_cat)-1))))
    code_cat = ['missing']+sorted(dx_multi.ICD9CM_CODE)*2+sorted(pr_multi.ICD9CM_CODE)+sorted(dx_ccs_cat)[1:]*2+sorted(pr_ccs_cat)[1:]
    n_code_cat = len(code_cat)    
    dx1_ccs_dict = dict(zip(dx_ccs_cat[1:], range(1+len(dx_multi)*2+len(pr_multi), len(dx_multi)*2+len(pr_multi)+len(dx_ccs_cat))))
    dx_ccs_dict = dict(zip(dx_ccs_cat[1:], range(1+len(dx_multi)*2+len(pr_multi)+len(dx_ccs_cat[1:]), 
                                                 1+len(dx_multi)*2+len(pr_multi)+len(dx_ccs_cat[1:])*2)))
    pr_ccs_dict = dict(zip(pr_ccs_cat[1:], range(1+len(dx_multi)*2+len(pr_multi)+len(dx_ccs_cat[1:])*2, n_code_cat)))
else:
    DX_dict = dict(zip(DX_cat, range(len(DX_cat))))
    PR_dict = dict(zip(PR_cat, [0] + list(range(len(DX_cat), len(DX_cat)+len(PR_cat)))))
    code_cat = ['missing'] + sorted(dx_multi.ICD9CM_CODE) + sorted(pr_multi.ICD9CM_CODE) + sorted(dx_ccs_cat)[1:] + sorted(pr_ccs_cat)[1:]
    n_code_cat = len(code_cat)
    dx_ccs_dict = dict(zip(dx_ccs_cat[1:], range(len(dx_multi)+len(pr_multi)+1, len(dx_multi)+len(pr_multi)+len(dx_ccs_cat))))
    pr_ccs_dict = dict(zip(pr_ccs_cat[1:], range(len(dx_multi)+len(pr_multi)+len(dx_ccs_cat), n_code_cat)))

parent_pairs = []
for dx in DX_cat[1:]:
    if not dx in unclassified:
        parent = dx_multi.loc[dx, 'finest_non_empty']
        if sep_dx1==1:
            parent_pairs.append((DX1_dict[dx], dx1_ccs_dict[parent]))
        parent_pairs.append((DX_dict[dx], dx_ccs_dict[parent]))
for pr in PR_cat[1:]:
    parent = pr_multi.loc[pr, 'finest_non_empty']
    parent_pairs.append((PR_dict[pr], pr_ccs_dict[parent]))
for c in dx_ccs_cat[1:]:
    parent = '.'.join(c.split('.')[:-1])
    if not parent=='':
        if sep_dx1==1:
            parent_pairs.append((dx1_ccs_dict[c], dx1_ccs_dict[parent]))
        parent_pairs.append((dx_ccs_dict[c], dx_ccs_dict[parent]))
for c in pr_ccs_cat[1:]:
    parent = '.'.join(c.split('.')[:-1])
    if not parent=='':
          parent_pairs.append((pr_ccs_dict[c], pr_ccs_dict[parent]))

cooccur_df = pd.read_csv(path+cooccur_file)

g = Glove(input_dim=len(code_cat), embedding_dim=embed_dim, count_cap=count_cap)
g.train_glove(cooccur_df=cooccur_df, cache_path=model_path+str(job_index)+'/', epochs=50, reducelr_patience=3, parent_pairs=parent_pairs, 
              lamb=lamb, metric=metric, verbose=2)
embed_mat = g.get_embed_mat()
bias_mat = g.get_bias_mat()

save_file = save_folder+'embed_mat_{0}_{1:.3f}_{2}_{3}{4}.npy'.format(embed_dim, lamb, count_cap, cohort, test_fold)
np.save(path+save_file, embed_mat)

save_file = save_folder+'bias_mat_{0}_{1:.3f}_{2}_{3}{4}.npy'.format(embed_dim, lamb, count_cap, cohort, test_fold)
np.save(path+save_file, bias_mat)
