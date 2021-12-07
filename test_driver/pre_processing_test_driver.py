#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:26:04 2021

@author: soms
"""

from statistical_tests import PreProcessing
from statistical_tests import LogitRegression

import pandas as pd

from sklearn.model_selection import train_test_split

obj = PreProcessing()


file = 'Default_On_Payment.csv'
data_org = pd.read_csv(file)



data = data_org.copy()


# 1.
obj.cols_to_lower_remove_spaces(data)

# print(data.columns)

# a close look at the data reveals that row # 4001 is corrupted, 
# and customerId is null,  so delete that row
# Delete those rown where customer_id is empty
df_del = data[data['customer_id'].isnull()]
if df_del.index is not None:
    data.drop(df_del.index, axis=0, inplace = True)

# cols_to_ignore = ['customer_id']
cols_to_ignore = []

target_var = 'default_on_payment'



# Divide columns into category and numeric data types by looking at the data-dictionary
cat_cols = ['status_checking_acc','credit_history','purposre_credit_taken','savings_acc',
            'years_at_present_employment','marital_status_gender','other_debtors_guarantors',
            'property','other_inst_plans','housing','job','telephone','foreign_worker','default_on_payment']

num_cols = ['duration_in_months','credit_amount','inst_rt_income','current_address_yrs','age','num_cc',
            'dependents']



obj.trigger_pre_processing(data, cat_cols, num_cols)





X = data.drop(target_var, axis=1)
y = data[target_var]

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3, random_state=100, stratify=y)

logit = LogitRegression(target_var, X,y)

logit.execute_logit('1st iteration')



