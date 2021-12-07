#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 11:43:13 2021

@author: soms
"""

from statistical_tests import LogitRegression
from statistical_tests import StatisticalAnalysis

import pandas as pd


# logit = LogitRegression('target_var',' X', 'y')



# logit.logit_process('X', 'y')







# file = 'CREDIT CARD USERS DATA.csv'
file = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df_original = pd.read_csv(file,low_memory=False)


df = df_original.copy()
target_var = 'Churn'
obj = StatisticalAnalysis(target_var)

#convert Churn to categorical value
# (df, cols_to_cat=[], cols_to_num=[])
obj.convert_dtypes(df,['Churn'],[])



df.dtypes



# Churn cols has values as 'Yes', 'No'. Convert those to 1,0
# pd.Series(map(lambda x: dict(Yes=1, No=0)[x],
#               df[target_var].values.tolist()), df.index)
# get_ipython().run_line_magic('timeit', '')
df[target_var] = df[target_var].apply(lambda x: 0 if x=='No' else 1)

df.head()
df.dtypes



obj.execute_statistical_tests(df)


obj.feature_lst