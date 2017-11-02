#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 13:29:15 2017

@author: leiding
"""

#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
%matplotlib inline


#getting data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape)
print(test.shape)
print("columns: "+ train.columns)


#checking data
train['SalePrice'].describe()
sns.distplot(train['SalePrice']);

#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


def plot_2column(dataset, var1,var2):
    data = pd.concat([dataset[var1], dataset[var2]], axis=1)
    data.plot.scatter(x=var2, y=var1, ylim=(0,800000))
    
    
    
plot_2column(train,'SalePrice', 'GrLivArea') 
plot_2column(train,'SalePrice', 'TotalBsmtSF')

plot_2column(train,'SalePrice', '1stFlrSF')


def plot_box(dataset, var1, var2):

    data = pd.concat([dataset[var1], dataset[var2]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var2, y=var1, data=data)
    fig.axis(ymin=0, ymax=800000);
    
    
plot_box(train,'SalePrice','OverallQual')    

plot_box(train,'SalePrice','YearBuilt')    
    
    
    
 #correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
    

#saleprice correlation matrix
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
    


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();







#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


#dealing with missing data
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max() #just checking that there's no missing data missing...



#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(df_train[train['Id'] == 1299].index)
train = train.drop(df_train[train['Id'] == 524].index)





#normalize some features
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)

train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)





#reduce some features
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

print(cols)

train_reduce = train[cols]
cols = cols[1:-1]
test_reduce = test[cols]


X = train_reduce.iloc[:,1:-1].values
y = train_reduce.iloc[:,0].values.reshape((y.shape[0],1))

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_cv = sc_X.transform(X_cv)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_cv = sc_y.transform(y_cv)





#fitting multile linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
result = regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_cv)


from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(regressor, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()




#building the optimal model using backward elimination
import statsmodels.formula.api as sm

X = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)
X_opt = X
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()









# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
"""
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
"""


from sklearn import model_selection
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)



# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


