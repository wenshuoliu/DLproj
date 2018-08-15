# The tools for multi-level CCS categories of ICD9 diagnosis(DX) and procedure(PR) codes
import pandas as pd

path = '/nfs/turbo/umms-awaljee/wsliu/Data/NRD/'

## DX codes labels
dx_label = pd.read_csv(path+'icd9dx2014.csv')

## Multi-level CCS categories of DX codes
dx_multi = pd.read_csv(path+'ccs_multi_dx_tool_2015.csv')
dx_multi.columns = ['ICD9CM_CODE', 'CCS_LVL1', 'CCS_LVL1_LABEL', 'CCS_LVL2', 'CCS_LVL2_LABEL', 'CCS_LVL3', 'CCS_LVL3_LABEL', 
                    'CCS_LVL4', 'CCS_LVL4_LABEL']
dx_multi.ICD9CM_CODE = dx_multi.ICD9CM_CODE.apply(lambda x:x.replace("'", "").replace(' ', ''))
for j in range(1, 5):
    dx_multi['CCS_LVL'+str(j)] = dx_multi['CCS_LVL'+str(j)].apply(lambda x:x.replace("'", "").replace(' ', ''))
    
## Multi-level CCS categories of PR codes
pr_multi = pd.read_csv(path+'ccs_multi_pr_tool_2015.csv')
pr_multi.columns = ['ICD9CM_CODE', 'CCS_LVL1', 'CCS_LVL1_LABEL', 'CCS_LVL2', 'CCS_LVL2_LABEL', 'CCS_LVL3', 'CCS_LVL3_LABEL']
pr_multi.ICD9CM_CODE = pr_multi.ICD9CM_CODE.apply(lambda x:x.replace("'", "").replace(' ', ''))
for j in range(1, 4):
    pr_multi['CCS_LVL'+str(j)] = pr_multi['CCS_LVL'+str(j)].apply(lambda x:x.replace("'", "").replace(' ', ''))