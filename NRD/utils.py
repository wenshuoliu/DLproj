import keras.backend as K
import pandas as pd
from ccs_tools import dx_multi, pr_multi

""" The regularization of the parent matrix """
class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Mat_reg(Regularizer):
    """Regularizer for parent matrix.
    # Arguments
        mat: numpy array; the matrix indicating the parent of each code.
        lamb: Float; the penalty tuning parameter. 
    """

    def __init__(self, mat, lamb, norm=2):
        self.lamb = K.cast_to_floatx(lamb)
        self.pmat = K.constant(value=mat, dtype=K.floatx(), name='parent_mat')
        self.norm = norm

    def __call__(self, embed_mat):
        diff = K.dot(self.pmat, embed_mat) #difference between each embedding and its parent
        if self.norm==2:
            return self.lamb*K.sum(K.square(diff))
        else:
            return self.lamb*K.sum(K.abs(diff))
        
        
class Parent_reg(Regularizer):
    """Regularizer by adding penalization between each embedding and its parent.
    # Arguments
        parent_pairs: list of tuples representing the parent of each code
        lamb: Float; the penalty tuning parameter. 
        metric: str, could be 'l1', 'l2', 'cosine'
    """

    def __init__(self, parent_pairs, lamb, metric='cosine'):
        self.lamb = K.cast_to_floatx(lamb)
        self.code_ind, self.parent_ind = zip(*parent_pairs)
        self.metric = metric

    def __call__(self, embed_mat):        
        #select the codes and their parents from the embedding matrix:
        embeds = K.gather(embed_mat, self.code_ind)
        parents = K.gather(embed_mat, self.parent_ind)
        if self.metric=='cosine':
            code_norm = K.sqrt(K.sum(K.square(embeds), axis=1))
            parent_norm = K.sqrt(K.sum(K.square(parents), axis=1))
            inner = K.sum(embeds*parents, axis=1)
            return self.lamb*(1.-K.mean(inner/(code_norm*parent_norm)))
        else:
            diff = embeds - parents
            if self.metric=='l2':
                return self.lamb*K.mean(K.square(diff))
            elif self.metric=='l1':
                return self.lamb*K.mean(K.abs(diff))
            else:
                raise ValueError('Metric should be l1, l2 or cosine.')

def get_frequency(data_df, subset=['DX1', 'DX', 'PR']):
    freq = dict()
    if 'DX1' in subset:
        freq['DX1'] = data_df['DX1'].value_counts()
    if 'DX' in subset:
        DXs = ['DX'+str(j) for j in range(2, 31)]
        DX_series = pd.concat([data_df[DX] for DX in DXs])
        freq['DX'] = DX_series.value_counts()
    if 'PR' in subset:
        PRs = ['PR'+str(j) for j in range(1, 16)]
        PR_series = pd.concat([data_df[PR] for PR in PRs])
        freq['PR'] = PR_series.value_counts()
    return freq

def preprocess(data_df, DX_rarecutpoint=0, PR_rarecutpoint=0):
    """
    This function takes a dataframe, converts all the codes and hospitals into integer ids. It also filters all the codes
    with frequency lower than the rarecutpoint. 
    """
    freq = get_frequency(data_df)
    unclassified = set(dx_multi.loc[dx_multi.CCS_LVL1 == '18', 'ICD9CM_CODE'])
    
    ## define the categories of DX1, DX, PR, dx1_ccs, dx_ccs and pr_ccs
    DX1_cat = sorted(freq['DX1'].index)
    DX_cat = sorted(freq['DX'].loc[(freq['DX']>DX_rarecutpoint) & (~freq['DX'].index.isin(unclassified)) & 
                               (~freq['DX'].index.isin(['invl', 'incn']))].index)
    PR_cat = sorted(freq['PR'].loc[(freq['PR']>PR_rarecutpoint) & (~freq['PR'].index.isin(['invl', 'incn']))].index)

    dx1_multi_sub = dx_multi.loc[dx_multi.ICD9CM_CODE.isin(DX1_cat)]
    dx1_ccs_cat = sorted(pd.concat([dx1_multi_sub['CCS_LVL'+str(j)] for j in range(1, 5)]).unique())
    if '' in dx1_ccs_cat:
        dx1_ccs_cat.remove('')
    dx_multi_sub = dx_multi.loc[dx_multi.ICD9CM_CODE.isin(DX_cat)]
    dx_ccs_cat = sorted(pd.concat([dx_multi_sub['CCS_LVL'+str(j)] for j in range(1, 5)]).unique())
    if '' in dx_ccs_cat:
        dx_ccs_cat.remove('')
    pr_multi_sub = pr_multi.loc[pr_multi.ICD9CM_CODE.isin(PR_cat)]
    pr_ccs_cat = sorted(pd.concat([pr_multi_sub['CCS_LVL'+str(j)] for j in range(1, 4)]).unique())
    if '' in pr_ccs_cat:
        pr_ccs_cat.remove('')
    code_cat = ['missing']+DX1_cat+DX_cat+PR_cat+dx1_ccs_cat+dx_ccs_cat+pr_ccs_cat
    
    ## define the dictionaries of the codes, and hospital to int
    DX1_dict = dict(zip(DX1_cat, range(1, len(DX1_cat)+1)))
    DX_dict = dict(zip(DX_cat, range(len(DX1_cat)+1, len(DX1_cat)+len(DX_cat)+1)))
    PR_dict = dict(zip(PR_cat, range(len(DX1_cat)+len(DX_cat)+1, len(DX1_cat)+len(DX_cat)+len(PR_cat)+1)))
    DX1_dict['missing'] = 0
    DX_dict['missing'] = 0
    PR_dict['missing'] = 0
    dx1_ccs_dict = dict(zip(dx1_ccs_cat, range(len(DX1_cat)+len(DX_cat)+len(PR_cat)+1, 
                                              len(DX1_cat)+len(DX_cat)+len(PR_cat)+len(dx1_ccs_cat)+1)))
    dx_ccs_dict = dict(zip(dx_ccs_cat, range(len(DX1_cat)+len(DX_cat)+len(PR_cat)+len(dx1_ccs_cat)+1, 
                                             len(DX1_cat)+len(DX_cat)+len(PR_cat)+len(dx1_ccs_cat)+len(dx_ccs_cat)+1)))
    pr_ccs_dict = dict(zip(pr_ccs_cat, range(len(DX1_cat)+len(DX_cat)+len(PR_cat)+len(dx1_ccs_cat)+len(dx_ccs_cat)+1, 
                                            len(DX1_cat)+len(DX_cat)+len(PR_cat)+len(dx1_ccs_cat)+len(dx_ccs_cat)+len(pr_ccs_cat)+1)))
    
    hosp_cat = sorted(data_df['HOSP_NRD'].value_counts().index)
    hosp_dict = dict(zip(hosp_cat, range(len(hosp_cat))))

    #### define parent_pairs to pre-train embedding with glove
    parent_pairs = []
    for dx in DX1_cat:
        parent = dx_multi.loc[dx, 'finest_non_empty']
        parent_pairs.append((DX1_dict[dx], dx1_ccs_dict[parent]))
    for dx in DX_cat:
        parent = dx_multi.loc[dx, 'finest_non_empty']    
        parent_pairs.append((DX_dict[dx], dx_ccs_dict[parent]))
    for pr in PR_cat:
        parent = pr_multi.loc[pr, 'finest_non_empty']
        parent_pairs.append((PR_dict[pr], pr_ccs_dict[parent]))
    for c in dx1_ccs_cat:
        parent = '.'.join(c.split('.')[:-1])
        if not parent=='':
            parent_pairs.append((dx1_ccs_dict[c], dx1_ccs_dict[parent]))
    for c in dx_ccs_cat:
        parent = '.'.join(c.split('.')[:-1])
        if not parent=='':
            parent_pairs.append((dx_ccs_dict[c], dx_ccs_dict[parent]))
    for c in pr_ccs_cat:
        parent = '.'.join(c.split('.')[:-1])
        if not parent=='':
            parent_pairs.append((pr_ccs_dict[c], pr_ccs_dict[parent]))
            
    n_DX = 29
    n_PR = 15
    DXs = ['DX'+str(n) for n in range(2, n_DX+2)]
    PRs = ['PR'+str(n) for n in range(1, n_PR+1)]

    int_df = data_df.copy()
    int_df.loc[~int_df.DX1.isin(DX1_cat), 'DX1'] = 'missing'
    int_df['DX1'] = int_df['DX1'].map(DX1_dict)
    
    for dx in DXs:
        int_df.loc[~int_df[dx].isin(DX_cat), dx] = 'missing'
        int_df[dx] = int_df[dx].map(DX_dict)
    
    for pr in PRs:
        int_df.loc[~int_df[pr].isin(PR_cat), pr] = 'missing'
        int_df[pr] = int_df[pr].map(PR_dict)
    
    int_df['HOSP_NRD'] = int_df['HOSP_NRD'].map(hosp_dict)
    
    return {'int_df':int_df, 'DX1_dict':DX1_dict, 'DX_dict':DX_dict, 'PR_dict':PR_dict, 'dx1_ccs_dict':dx1_ccs_dict, 
            'dx_ccs_dict':dx_ccs_dict, 'pr_ccs_dict':pr_ccs_dict, 'hosp_dict':hosp_dict, 'parent_pairs':parent_pairs, 
           'code_cat':code_cat, 'DX1_cat':DX1_cat, 'DX_cat':DX_cat, 'PR_cat':PR_cat, 'hosp_cat':hosp_cat}
                
                
