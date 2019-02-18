import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--DX1_dim', type=int, default=10)
parser.add_argument('--DX_dim', type=int, default=100)
parser.add_argument('--PR_dim', type=int, default=50)
parser.add_argument('--penalty', type=float, default=1.)
parser.add_argument('--penalty_metric', type=str, default='l2')
parser.add_argument('--count_cap', type=int, default=100)
parser.add_argument('--dx1_rarecutpoint', type=int, default=10)
parser.add_argument('--dx_rarecutpoint', type=int, default=10)
parser.add_argument('--pr_rarecutpoint', type=int, default=10)
parser.add_argument('--save_folder', type=str, default='all/sepdx1/')

parser.add_argument('--job_index', type=int, default=0)

args = parser.parse_args()
DX1_dim = args.DX1_dim
DX_dim = args.DX_dim
PR_dim = args.PR_dim
penalty = args.penalty
penalty_metric = args.penalty_metric
count_cap = args.count_cap
DX1_rarecutpoint = args.dx1_rarecutpoint
DX_rarecutpoint = args.dx_rarecutpoint
PR_rarecutpoint = args.pr_rarecutpoint
save_folder = args.save_folder

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
model_path = path + 'models/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)
    
from glove import GloveMS
from ccs_tools import dx_multi, pr_multi, core_dtypes_pd
from utils import Parent_reg, get_frequency, preprocess, preprocess_ms

unclassified = set(dx_multi.loc[dx_multi.CCS_LVL1 == '18', 'ICD9CM_CODE'])
folder = 'elder/'
DX1_freq = pd.read_csv(path+folder+'DX1_freq.csv', dtype={'DX1':str, 'frequency':int})
DX_freq = pd.read_csv(path+folder+'DX_freq.csv', dtype={'DX':str, 'frequency':int})
PR_freq = pd.read_csv(path+folder+'PR_freq.csv', dtype={'PR':str, 'frequency':int})
DX1_freq = DX1_freq.loc[DX1_freq.frequency>0]
DX_freq = DX_freq.loc[DX_freq.frequency>0]
PR_freq = PR_freq.loc[PR_freq.frequency>0]
DX1_cat = sorted(DX1_freq.loc[(DX1_freq.frequency>DX1_rarecutpoint) & (~DX1_freq.DX1.isin(unclassified))].DX1)
DX_cat = sorted(DX_freq.loc[(DX_freq.frequency>DX_rarecutpoint) & (~DX_freq.DX.isin(unclassified))].DX)
PR_cat = sorted(PR_freq.loc[(PR_freq.frequency>PR_rarecutpoint)].PR)
n_DX1_cat = len(DX1_cat)
n_DX_cat = len(DX_cat)
n_PR_cat = len(PR_cat)

cooccur_all = pd.read_csv(path+folder+'cooccur_df{}.csv'.format(DX1_rarecutpoint))
all_df = pd.read_csv(path+folder+'cohorts10/ami/ami_pred.csv', dtype=core_dtypes_pd)

preprocessed_ms = preprocess_ms(all_df, DX1_cat=DX1_cat, DX_cat=DX_cat, PR_cat=PR_cat)
dx1_ccs_dict = preprocessed_ms['dx1_ccs_dict']
dx_ccs_dict = preprocessed_ms['dx_ccs_dict']
pr_ccs_dict = preprocessed_ms['pr_ccs_dict']
dx1_parent_pairs = preprocessed_ms['dx1_parent_pairs']
dx_parent_pairs = preprocessed_ms['dx_parent_pairs']
pr_parent_pairs = preprocessed_ms['pr_parent_pairs']

gms = GloveMS(n_DX1_cat=n_DX1_cat, n_DX_cat=n_DX_cat, n_PR_cat=n_PR_cat, n_dx1_ccs=len(dx1_ccs_dict), 
                  n_dx_ccs=len(dx_ccs_dict), n_pr_ccs=len(pr_ccs_dict), DX1_dim=DX1_dim, DX_dim=DX_dim, PR_dim=PR_dim, 
                  count_cap=count_cap, scaling_factor=0.75)
gms.train_glove(cooccur_df=cooccur_all, cache_path=model_path+'temp/{}/'.format(job_index), dx1_parent_pairs=dx1_parent_pairs, 
                    dx_parent_pairs=dx_parent_pairs, pr_parent_pairs=pr_parent_pairs, lamb=penalty, metric=penalty_metric, verbose=2, 
                   batch_size=1024*4, epochs=100, earlystop_patience=10, reducelr_patience=2)
embeds = gms.get_embed_mat()

DX1_embed = embeds['DX1_embed']
DX_embed = embeds['DX_embed']
PR_embed = embeds['PR_embed']

params = '{}_{}_{}_{}{:.1f}_{}'.format(DX1_dim, DX_dim, PR_dim, penalty_metric, penalty, count_cap)
np.savez(path+save_folder+str(DX1_rarecutpoint)+'/embed_mats_'+params+'.npz', DX1_embed=DX1_embed, DX_embed=DX_embed, PR_embed=PR_embed)
