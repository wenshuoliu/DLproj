import numpy as np
import pandas as pd
import os, sys
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import statsmodels.stats.api as sms

model_names = ['demo', 'demo_dx1', 'demo_dx1_dx', 'demo_dx1_dx_pr']
code_embed_dims = [200]
fc_widths = [512]
md_widths = [128]
lr1s = [2e-4]
lr2s = [2e-5]
dropouts = [0.3]
batchsizes = [256]
penalties = [0.]
count_caps = [20]
tst_seeds = range(10)
cohorts = ['ami']
zips = [(embed_dim, tst_fold, cohort, 
         'all/sepdx1/test_embed/cosine/embed_mat_{0}_{1:.3f}_{2}_{3}{4}.npy'.format(embed_dim, penalty, count_cap, cohort, 
                                                                                    tst_fold)) 
        for embed_dim in code_embed_dims for penalty in penalties for count_cap in count_caps for cohort in cohorts 
        for tst_fold in tst_seeds] 
#zips = zips + [(embed_dim, tst_fold, '') for embed_dim in code_embed_dims for tst_fold in tst_seeds]
sep_dx1s = [1]
val_folds = [7]
rho_widths = [16]
result_files = ['output/ht_result1123_{}.csv']

para_itr = itertools.product(model_names, fc_widths, md_widths, lr1s, lr2s, dropouts, batchsizes, zips, sep_dx1s, 
                             val_folds, rho_widths, result_files)

para_lst = [(mo, z[0], fc, md, lr1, lr2, dr, ba, z[3], z[1], z[2], se, va, rw, re) for mo, fc, md, lr1, lr2, dr, ba, z, se, va, rw, re in para_itr]

n_jobs = 10
for para, job_ind in zip(para_lst, itertools.cycle(range(n_jobs))):
    with open('hypertune'+str(job_ind)+'.sh', 'a') as f:
        f.write('python train_template_all0827.py --model_name {} --code_embed_dim {} --fc_width {} --md_width {} --lr1 {} --lr2 {} --dropout {} --batchsize {} --embed_file {} --tst_seed {} --cohort {} --sep_dx1 {} --val_fold {} --rho_width {} --result_file {} --job_index {}\n'.format(*para, job_ind))
