"""
#----------------------------------------------------------------------------------+
#      This python client gives you an ability to:                                 |
#       - Perform transformations on the macro-economic variables                  |
#----------------------------------------------------------------------------------+
"""
__author__ = "James Mc Neill"
__version__ = "1.0"
__maintainer__ = "James Mc Neill"
__email__ = "jmcneill06@outlook.com"
__status__ = "Test"

# Import packages
import pandas as pd
import numpy as np
from scipy.stats import norm

class Transformations:
    '''
    Perform time series variable transformations on the macro-economic variables
    '''
        
    # Method - Create UDF for the basic transformations
    def basic_transformations(self, df, addTrans=None):
        """
        Basic Transformations
        > - _R Raw series
        > - _Y y/y% changes
        > - _Q q/q% changes
        > - _D 1st difference
        > - _S 4th difference (seasonal difference)
        Additional Transformations (if required)
        > - _L logit transformation
        > - _X 1st difference of logit
        ***
        - ToDo
        > - _M MA smoothing and then y/y% changes
        > - _G 1st difference of y/y% changes
        > - _J 1st difference of q/q% changes
        > - _B value at time t / value at time 2 years previous
        > - _P probit transformation
        > - _Z 1st difference of probit (only if probit requested)
        """
        for col in df.columns:
            # Raw Series
            df[col+'_R'] = df[col]
            # y/y% changes
            df[col+'_Y'] = df[col].pct_change(4)
            # q/q% changes
            df[col+'_Q'] = df[col].pct_change()
            # 1st difference
            df[col+'_D'] = df[col].diff()
            # 4th difference
            df[col+'_S'] = df[col].diff(4)
            # Completing the additional transformations
            if addTrans == 1:
                #pass
                # Logit transformations
                df[col+'_L'] = np.log(df[col] / (1 - df[col]))
                df[col+'_X'] = df[col+'_L'].diff()
                # Probit transformations
                df[col+'_P'] = norm.ppf(df[col])
                df[col+'_Z'] = df[col+'_P'].diff()
        return df
    
    # Method - Create UDF for the creation of lags
    def create_lags(self, df, maxLag=4):
        for col in df.columns:
            for lag in np.arange(0, maxLag + 1):
                df[col+'_L'+str(lag)] = df[col].shift(lag)
        return df
    
    # Method - Test for the number of missing values in a column transformation
    def missing_infinity_values(self, df):
        df_out = pd.DataFrame(columns=['Column', 'MissingVals', 'InfinityVals'])
        for col in df.columns:
            # Check for missing values
            df_out = df_out.append(
                {"Column" : col,
                 "MissingVals" : df[col].isnull().sum(),
                 "InfinityVals" : np.isinf(df[col]).values.sum()
                }
            ,ignore_index=True)
        return df_out
