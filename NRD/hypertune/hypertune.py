import numpy as np
import pandas as pd
import os, sys
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import statsmodels.stats.api as sms

model_names = ['setsum_nn']
code_embed_dims = [200]
fc_widths = [512]
md_widths = [128]
lr1s = [1e-4, 2e-4, 5e-4]
#lr2s = [2e-5]
dropouts = [0.3]
batchsizes = [512, 1024]
embed_mats = ['random']
#penalties = [0, 0.5, 1.]
#penalty_metrics = ['cosine']
#count_caps = [0, 5, 20]
tst_seeds = [0]
cohorts = ['ami']
#DX_rarecutpoints = [20]
#PR_rarecutpoints = [drp/2 for drp in DX_rarecutpoints]
val_seeds = [0, 1, 2]
lambs = [0.2, 1., 10.]
n_sampless = [100]
result_files = ['output/ht_result1210_{}.csv']

para_itr = itertools.product(model_names, code_embed_dims, fc_widths, md_widths, lr1s, dropouts, batchsizes, embed_mats, 
                            tst_seeds, cohorts, val_seeds, lambs, n_sampless, result_files)
para_lst = [(mn, ced, fc, md, l1, do, bs, em, ts, ch, vs, la, ns, rf) 
            for mn, ced, fc, md, l1, do, bs, em, ts, ch, vs, la, ns, rf in para_itr]

n_jobs = 10
for para, job_ind in zip(para_lst, itertools.cycle(range(n_jobs))):
    with open('hypertune'+str(job_ind)+'.sh', 'a') as f:
        f.write('python template_skipgram1209.py --model_name {} --code_embed_dim {} --fc_width {} --md_width {} --lr1 {} --dropout {} --batchsize {} --embed_file {} --tst_seed {} --cohort {} --val_seed {} --lamb {} --n_samples {} --result_file {} --job_index {}\n'.format(*para, job_ind))
