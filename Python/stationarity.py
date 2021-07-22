"""
#----------------------------------------------------------------------------------+
#      This python client gives you an ability to:                                 |
#       - Perform stationarity testing on the macro-economic variables             |
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
from statsmodels.tsa.stattools import adfuller, kpss

class Stationarity:
    '''
    Perform stationarity testing on a list of macro-economic variables
    '''
    # Constructor
    #def __init__(self):
       
    # Method - Create UDF for the basic transformations
    def stationarity_test(self, df):
        # Construct empty DataFrame with required output columns
        tmp = pd.DataFrame(columns=['variable', 'obs', 'adfstat',
                                    'adfpvalue', 'kpssstat', 'kpsspvalue',
                                    'adf_stat', 'kpss_stat'
                                   ])
        for col in df.columns:
            # Keep only the not null values for review - method requires a series input
            result = adfuller(df.loc[df[col].notnull(),col], autolag='AIC')
            stat = result[0]
            pval = result[1]

            # KPSS test. The option 'ct' for the regression parameter, means that the 'deterministic trend' is reviewed
            # instead of the mean value
            stats, p, lags, cv = kpss(df.loc[df[col].notnull(),col], 'ct', nlags='auto')

            # Populate the tmp DataFrame
            tmp = tmp.append({'variable': col
                              ,'obs': len(df.loc[df[col].notnull(),col])
                              ,'adfstat': stat
                              ,'adfpvalue': pval
                              ,'kpssstat': stats
                              ,'kpsspvalue': p
                              ,'adf_stat': np.where(pval > 0.05, 
                                                    'Non-Stationary',
                                                    'Stationary')
                              ,'kpss_stat': np.where(p < 0.05,
                                                    'Non-Stationary',
                                                    'Stationary')
                             }
                            ,ignore_index=True)
        return tmp
