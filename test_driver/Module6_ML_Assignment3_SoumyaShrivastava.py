# #### https://www.quora.com/What-exactly-is-a-logistic-regression-algorithm-in-machine-learning-What-are-its-applications/answer/Ratnakar-Pandey-RP

# * EDA pre data cleaning and insights
# * EDA post data cleaning and insights
# * Build a linear regression model and evaluate model performance
# * Which variables are the most important to predict the label
# * Would you recommend you clients to use this model, why or why not?
# * Create a Power Point presentation and present in the next class


from statistical_tests import LogitRegression
from statistical_tests import DecisionTree
from statistical_tests import RandomForest



#Import Required libraries 
import numpy as np
import pandas as pd
import seaborn as sns                                             
import pandas_profiling
# import tkinter




import sklearn
from sklearn import datasets

# get_ipython().run_line_magic('matplotlib', 'inline')
import tkinter
import matplotlib.pyplot as plt 
# plt.use('TkAgg')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.metrics import roc_curve,auc




#Print multiple statements in same line

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'



file = 'Default_On_Payment.csv'
data_org = pd.read_csv(file)



data = data_org.copy()


# # 1. EDA and insights before cleaning



data.describe()


data.head()



data.columns = data.columns.str.strip()
data.columns = [str(x).lower() for x in data.columns]

data.head()


print(data.dtypes)



corr_matrix = data.corr()
plt.subplots(figsize= (16,10 ))
sns.heatmap(corr_matrix, cmap='RdYlGn', annot= True)
plt.show()



# eda_report = pandas_profiling.ProfileReport(data)



# eda_report






# # 2. Data cleaning and EDA and insights post cleaning

# # Data cleaning process starts


# Anamolies in file
# Delete those rown where customer_id is empty


df_del = data[data['customer_id'].isnull()]
if df_del.index is not None:
    data.drop(df_del.index, axis=0, inplace = True)





# ## 2.1 Divide data into categorical and numerical columns




# columns which can be ignored
cols_to_ignore = []
target_var = 'default_on_payment'



# Divide columns into category and numeric data types by looking at the data-dictionary
cat_cols = ['status_checking_acc','credit_history','purposre_credit_taken','savings_acc',
            'years_at_present_employment','marital_status_gender','other_debtors_guarantors',
            'property','other_inst_plans','housing','job','telephone','foreign_worker','default_on_payment']

num_cols = ['duration_in_months','credit_amount','inst_rt_income','current_address_yrs','age','num_cc',
            'dependents']



# Remove cols not required
if (set(cols_to_ignore).issubset(data.columns)):
    data.drop( cols_to_ignore, axis = 1, inplace = True )

# convert cat_cols to Categories explicitly
for col in cat_cols:
    data[col] = pd.Categorical(data[col])

# convert num_cols to numeric explicitly
for col_ in num_cols:
    data[col_] = pd.to_numeric(data[col_]) 
    
    


# ## 2.2 Missing value, Null value treatment

# In[182]:


# # Data cleaning
data.isnull().sum()
print('--')

data.isna().sum()


# In[183]:


data[data.isnull().any(axis=1)]


# NaN treatment

for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

for col_ in num_cols:
    data[col_] = data[col_].fillna(data[col_].median())   



# # Data cleaning
data.isnull().sum()
print('--')

data.isna().sum()


# ## 2.3 Duplicate treatment

data.duplicated().sum()
print('--')



data.drop_duplicates( subset = None, keep = 'first', inplace = True)



data.duplicated().sum()
print('--')


# ## 2.4. Identifying Multicollinearity with numerical variables
# 
# Multicollinearity occurs when independent variables in a regression model are correlated


# Not imp for logit regression. Imp for linear regression 


# Identifying Multicollinearity with numerical variables

# from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Correlation is often used to find the relationship ( both strength and direction) between a feature and the target:
# 2. Collinearity, on the other hand, is a situation where two features are linearly associated (high correlation),
# and they are used as predictors for the target.
# 3. Multicollinearity is a special case of collinearity where a feature exhibits a linear relationship with 
# two or more features.

# VIF allows you to determine the strength of the correlation between the various independent variables. 
# It is calculated by taking a variable and regressing it against every other variables.


# VIF value intrepretition 

# VIF = 1 — features are not correlated ie No correlation
# 1<VIF<5 — features are moderately correlated ie Moderate correlation
# VIF>5 — features are highly correlated
# VIF>10 — high correlation between features and is cause for concern






from statsmodels.stats.outliers_influence import variance_inflation_factor

def get_vif_scores(df , target_var):
    
    if target_var in df.columns:
        df = df.drop(target_var, axis=1)

    vif_scores = pd.DataFrame()
    vif_scores["features"] = df.columns
    vif_scores["vif_scores"] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
    vif_scores.sort_values('vif_scores', ascending = False, inplace=True)

    return vif_scores


# choose olnly features and also remove target variable
# call function
df_nums = data[num_cols]

df_vif = get_vif_scores(df_nums, target_var )

df_to_drop = df_vif.loc[df_vif['vif_scores']>10]

cols_to_drop  = df_to_drop['features']

print('\nTheoritically but not dropping: cols_to_drop::',cols_to_drop.tolist())

# Note : from above list drop the one with max VIF and then run the multicollinearity test again

# WIP

# Don't forget to update num_col list by deleting the element ( cols) deleted in above step from cols_to_drop

# num_cols = [x for x in num_cols if x not in cols_to_drop.tolist()]

num_cols





corr_matrix = data[num_cols].corr()
plt.subplots(figsize= (16,10 ))
sns.heatmap(corr_matrix, cmap='RdYlGn', annot= True)
plt.show()


# # 2.5 Coorelation 
# linear relationship between independent/target variable and feature variables


cols_temp = num_cols.copy()
cols_temp.append(target_var)
cols_temp
data[cols_temp].corr()
num_cols



# import seaborn as sns
# sns.pairplot(data[num_cols])


# ## 2.5 Outlier treatment


#Before outlier-treatment

data.describe()

data.hist(alpha=0.5, figsize=(20, 10))
plt.tight_layout()
plt.show()

data[target_var].hist();
data.groupby(target_var).size()


# Distribution plot only for numerical columns, don't use cat cols which were encoded or which are Categorical 

# Before outlier-treatment 
for col in num_cols:
    data.boxplot(col)
    plt.show()



# between 1st and 99th percentile
# treat all columns as all are in numerical form now

# Verify Check with Prof
# quantile should be decided based on the extent of outliers per col 
# or for all cols in one go?
# Prof:: same filtering can be generic across features or 5 , 95 percentile
for x in num_cols:
    outlier = data[x].quantile([0.10,0.90]).values    
    data[x] = np.clip(data[x],outlier[0],outlier[1])


# In[196]:


# Post outlier-treatment 

data.describe()

data.hist(alpha=0.5, figsize=(20, 10))
plt.tight_layout()
plt.show()

# data[target_var].hist();
# data.groupby(target_var).size()




# Distribution plot only for numerical columns, don't use cat cols which were encoded

# Post outlier-treatment 

for col in num_cols:
    data.boxplot(col)
    plt.show()



for col in num_cols:
    data.plot.scatter(x=col, y=target_var)
    plt.xlabel(col, fontsize=18)
    plt.ylabel(target_var, fontsize=18)
    plt.show()


# ## 2.6 Dummy Coding - only for Categorical features
# process of coding a categorical variable into dichotomous variables ie variable with only two possible values


# Dummy encoding, include target_var as well
for x in cat_cols:
    #if x == target_var: 
        #continue
    data[x] = pd.Categorical(data[x]).codes



d1 = data.where(data['foreign_worker']=='Yes')

len(d1)


# ## 2.7 Min Max Scaler - only for Numerical features


from sklearn import preprocessing



# Before Normalization
data.head()



# Random test : Comparision of 3 Scaling Algos
scaler = preprocessing.RobustScaler()
robust_df = scaler.fit_transform(data)
robust_df = pd.DataFrame(robust_df, columns = data.columns)
 
scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(data)
standard_df = pd.DataFrame(standard_df, columns = data.columns)
 
scaler = preprocessing.MinMaxScaler()
minmax_df = scaler.fit_transform(data)
minmax_df = pd.DataFrame(minmax_df, columns = data.columns)




# for col in data.columns:

#     col
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, figsize =(20, 5))


#     ax1.set_title('Before Scaling')
#     ax1.xlabel=col

#     sns.kdeplot(data[col], ax = ax1, color ='r')
#     ax2.set_title('After Robust Scaling')
#     ax2.xlabel=col

#     sns.kdeplot(robust_df[col], ax = ax2, color ='red')
#     ax3.set_title('After Standard Scaling')
#     ax3.xlabel=col

#     sns.kdeplot(standard_df[col], ax = ax3, color ='black')
#     ax4.set_title('After Min-Max Scaling')
#     ax4.xlabel=col

#     sns.kdeplot(minmax_df[col], ax = ax4, color ='black')
#     plt.show()

    
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, figsize =(20, 5))    

# sns.kdeplot(data['credit_amount'], ax = ax1, color ='r')
# sns.kdeplot(data['duration_in_months'], ax = ax1, color ='b')
# ax2.set_title('After Robust Scaling')
 
# sns.kdeplot(robust_df['credit_amount'], ax = ax2, color ='red')
# sns.kdeplot(robust_df['duration_in_months'], ax = ax2, color ='blue')
# ax3.set_title('After Standard Scaling')
 
# sns.kdeplot(standard_df['credit_amount'], ax = ax3, color ='black')
# sns.kdeplot(standard_df['duration_in_months'], ax = ax3, color ='g')
# ax4.set_title('After Min-Max Scaling')
 
# sns.kdeplot(minmax_df['credit_amount'], ax = ax4, color ='black')
# sns.kdeplot(minmax_df['duration_in_months'], ax = ax4, color ='g')
# plt.show()



#scld = MinMaxScaler()

# By using RobustScaler(), we can remove the outliers 
scld = preprocessing.RobustScaler()

# only for numerical columns. but categorical are also converted to numeric becuase of dummy coding
# tip to run scelar only for cols 'x' and 'z'
# df[['x','z']] = minmax_scale(df[['x','z']])

data_transformed = scld.fit_transform(data) # feature_range = (0,1)
scld_df_org = pd.DataFrame(data_transformed, columns = data.columns)



scld_df = scld_df_org.copy()
scld_df.head()
scld_df.describe()



#1. Categorical df 
df_cat = data[cat_cols]
# 2. Numerical df: create seperate df
df_num = data[num_cols]


# df_num.head()
# df_cat.head()


# Let’s try to remove these features one by one and observe their new VIF values.

# df_X.drop('credit_history', axis=1, inplace=True)

# df_X.drop('job', axis=1, inplace=True) 

# df_X.drop('marital_status_gender', axis=1, inplace=True) 




df_vif = get_vif_scores(scld_df, target_var )

df_to_drop = df_vif.loc[df_vif['vif_scores']>3]
# df_to_drop

cols_to_drop  = df_to_drop['features']

print('\ncols_to_drop::',cols_to_drop.tolist())


# # 3. Carry out Statistical Test analysis


X = scld_df.drop(target_var, axis=1)
y = scld_df[target_var]
corr_threshold  = 0.001





# X.head()
# y.head()
# train and test 
#1. get train and test data
#Split the data into training and test data (70/30 ratio)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3, random_state=100, stratify=y)


#2. fit model 
#fit the logisitc regression model on training dataset 

logit = LogitRegression(target_var, X,y)


coeff_threshold = 0.005

res_1 = logit.execute_logit('model improvement/feature selection')


# # # Assigment 3

# # # Decision Tree




dt = DecisionTree(target_var, X,y)

dt.execute_decision_tree(scld_df)




# Random Forest
rf = RandomForest(target_var, X,y)
rf.exec_ranfom_forest()




