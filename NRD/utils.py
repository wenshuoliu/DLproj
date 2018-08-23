import keras.backend as K

# dtypes of the core file, for reading from csv file:
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

core_cols = ['AGE', 'AWEEKEND', 'DIED', 'DISCWT', 'DISPUNIFORM', 'DMONTH', 'DQTR', 'DRG', 'DRGVER', 'DRG_NoPOA', 'DX1', 'DX2', 'DX3', 'DX4', 'DX5', 'DX6', 'DX7', 'DX8', 'DX9', 'DX10', 'DX11', 'DX12', 'DX13', 'DX14', 'DX15', 'DX16', 'DX17', 'DX18', 'DX19', 'DX20', 'DX21', 'DX22', 'DX23', 'DX24', 'DX25', 'DX26', 'DX27', 'DX28', 'DX29', 'DX30', 'DXCCS1', 'DXCCS2', 'DXCCS3', 'DXCCS4', 'DXCCS5', 'DXCCS6', 'DXCCS7', 'DXCCS8', 'DXCCS9', 'DXCCS10', 'DXCCS11', 'DXCCS12', 'DXCCS13', 'DXCCS14', 'DXCCS15', 'DXCCS16', 'DXCCS17', 'DXCCS18', 'DXCCS19', 'DXCCS20', 'DXCCS21', 'DXCCS22', 'DXCCS23', 'DXCCS24', 'DXCCS25', 'DXCCS26', 'DXCCS27', 'DXCCS28', 'DXCCS29', 'DXCCS30', 'ECODE1', 'ECODE2', 'ECODE3', 'ECODE4', 'ELECTIVE', 'E_CCS1', 'E_CCS2', 'E_CCS3', 'E_CCS4', 'FEMALE', 'HCUP_ED', 'HOSP_NRD', 'KEY_NRD', 'LOS', 'MDC', 'MDC_NoPOA', 'NCHRONIC', 'NDX', 'NECODE', 'NPR', 'NRD_DaysToEvent', 'NRD_STRATUM', 'NRD_VisitLink', 'ORPROC', 'PAY1', 'PL_NCHS', 'PR1', 'PR2', 'PR3', 'PR4', 'PR5', 'PR6', 'PR7', 'PR8', 'PR9', 'PR10', 'PR11', 'PR12', 'PR13', 'PR14', 'PR15', 'PRCCS1', 'PRCCS2', 'PRCCS3', 'PRCCS4', 'PRCCS5', 'PRCCS6', 'PRCCS7', 'PRCCS8', 'PRCCS9', 'PRCCS10', 'PRCCS11', 'PRCCS12', 'PRCCS13', 'PRCCS14', 'PRCCS15', 'PRDAY1', 'PRDAY2', 'PRDAY3', 'PRDAY4', 'PRDAY5', 'PRDAY6', 'PRDAY7', 'PRDAY8', 'PRDAY9', 'PRDAY10', 'PRDAY11', 'PRDAY12', 'PRDAY13', 'PRDAY14', 'PRDAY15', 'REHABTRANSFER', 'RESIDENT', 'SAMEDAYEVENT', 'SERVICELINE', 'TOTCHG', 'YEAR', 'ZIPINC_QRTL']

core_labels = ['Age in years at admission', 'Admission day is a weekend', 'Died during hospitalization', 'Weight to discharges in AHA universe', 'Disposition of patient (uniform)', 'Discharge month', 'Discharge quarter', 'DRG in effect on discharge date', 'DRG grouper version used on discharge date', 'DRG in use on discharge date, calculated without POA', 'Diagnosis 1', 'Diagnosis 2', 'Diagnosis 3', 'Diagnosis 4', 'Diagnosis 5', 'Diagnosis 6', 'Diagnosis 7', 'Diagnosis 8', 'Diagnosis 9', 'Diagnosis 10', 'Diagnosis 11', 'Diagnosis 12', 'Diagnosis 13', 'Diagnosis 14', 'Diagnosis 15', 'Diagnosis 16', 'Diagnosis 17', 'Diagnosis 18', 'Diagnosis 19', 'Diagnosis 20', 'Diagnosis 21', 'Diagnosis 22', 'Diagnosis 23', 'Diagnosis 24', 'Diagnosis 25', 'Diagnosis 26', 'Diagnosis 27', 'Diagnosis 28', 'Diagnosis 29', 'Diagnosis 30', 'CCS: diagnosis 1', 'CCS: diagnosis 2', 'CCS: diagnosis 3', 'CCS: diagnosis 4', 'CCS: diagnosis 5', 'CCS: diagnosis 6', 'CCS: diagnosis 7', 'CCS: diagnosis 8', 'CCS: diagnosis 9', 'CCS: diagnosis 10', 'CCS: diagnosis 11', 'CCS: diagnosis 12', 'CCS: diagnosis 13', 'CCS: diagnosis 14', 'CCS: diagnosis 15', 'CCS: diagnosis 16', 'CCS: diagnosis 17', 'CCS: diagnosis 18', 'CCS: diagnosis 19', 'CCS: diagnosis 20', 'CCS: diagnosis 21', 'CCS: diagnosis 22', 'CCS: diagnosis 23', 'CCS: diagnosis 24', 'CCS: diagnosis 25', 'CCS: diagnosis 26', 'CCS: diagnosis 27', 'CCS: diagnosis 28', 'CCS: diagnosis 29', 'CCS: diagnosis 30', 'E code 1', 'E code 2', 'E code 3', 'E code 4', 'Elective versus non-elective admission', 'CCS: E Code 1', 'CCS: E Code 2', 'CCS: E Code 3', 'CCS: E Code 4', 'Indicator of sex', 'HCUP Emergency Department service indicator', 'NRD hospital identifier', 'NRD record identifier', 'Length of stay (cleaned)', 'MDC in effect on discharge date', 'MDC in use on discharge date, calculated without POA', 'Number of chronic conditions', 'Number of diagnoses on this record', 'Number of E codes on this record', 'Number of procedures on this record', 'Timing variable used to identify days between admissions', 'NRD stratum used for weighting', 'NRD visitlink', 'Major operating room procedure indicator', 'Primary expected payer (uniform)', 'Patient Location: NCHS Urban-Rural Code', 'Procedure 1', 'Procedure 2', 'Procedure 3', 'Procedure 4', 'Procedure 5', 'Procedure 6', 'Procedure 7', 'Procedure 8', 'Procedure 9', 'Procedure 10', 'Procedure 11', 'Procedure 12', 'Procedure 13', 'Procedure 14', 'Procedure 15', 'CCS: procedure 1', 'CCS: procedure 2', 'CCS: procedure 3', 'CCS: procedure 4', 'CCS: procedure 5', 'CCS: procedure 6', 'CCS: procedure 7', 'CCS: procedure 8', 'CCS: procedure 9', 'CCS: procedure 10', 'CCS: procedure 11', 'CCS: procedure 12', 'CCS: procedure 13', 'CCS: procedure 14', 'CCS: procedure 15', 'Number of days from admission to PR1', 'Number of days from admission to PR2', 'Number of days from admission to PR3', 'Number of days from admission to PR4', 'Number of days from admission to PR5', 'Number of days from admission to PR6', 'Number of days from admission to PR7', 'Number of days from admission to PR8', 'Number of days from admission to PR9', 'Number of days from admission to PR10', 'Number of days from admission to PR11', 'Number of days from admission to PR12', 'Number of days from admission to PR13', 'Number of days from admission to PR14', 'Number of days from admission to PR15', 'A combined record involving rehab transfer', 'Patient State is the same as Hospital State', 'Transfer flag indicating combination of discharges involve same day events', 'Hospital Service Line', 'Total charges (cleaned)', 'Calendar year', 'Median household income national quartile for patient ZIP Code']

core_dtypes_sas = ['int', 'int', 'int', 'float', 'int', 'int', 'int', 'int', 'int', 'int', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'str', 'str', 'str', 'str', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'str', 'int', 'int', 'int', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'str', 'int', 'int', 'int', 'int']

core_dtypes_logic = ['continuous', 'categorical', 'categorical', 'continuous', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'embedding', 'embedding', 'embedding', 'embedding', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'continuous', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'continuous', 'categorical', 'embedding', 'categorical', 'categorical', 'categorical', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'embedding', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'categorical', 'categorical', 'categorical', 'categorical', 'continuous', 'categorical', 'categorical']

na_values = ['-9', '-8', '-6', '-5', '-99', '-88', '-66', '-999', '-888', '-666', '-9.9', '-8.8', '-6.6', '-9999', '-8888', '-6666', '-9.99', '-8.88', '-6.66', '-99999', '-88888', '-66666', '-99.99', '-88.88', '-66.66', '-999.99', '-888.88', '-666.66', '-9999999', '-8888888', '-6666666', '-9999.99', '-8888.88', '-6666.66', '-99.9999', '-88.8888', '-66.6666', '-999999999', '-888888888', '-666666666', '-9999.9999', '-8888.8888', '-6666.6666', '-999.99999', '-888.88888', '-666.66666', '-999999999', '-888888888', '-666666666', '-99.9999999', '-88.8888888', '-66.6666666', '-99999999.99', '-88888888.88', '-66666666.66', '-99999.99999', '-88888.88888', '-66666.66666', '-999999999999', '-888888888888', '-666666666666', '-99999999999.99', '-88888888888.88', '-66666666666.66']

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
