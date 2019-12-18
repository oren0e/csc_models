from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

# Read data
data = arff.loadarff('./data/caesarian.csv.arff')
df = pd.DataFrame(data[0])

# convert from bytes to int
for col in df.columns:
    df[col] = df[col].astype(int)

df.info()
df.head()
df.describe()

# EDA #
df.isnull().values.any()  # no missing values

sns.pairplot(df)
plt.show()

plt.style.use('ggplot')
plt.figure(figsize=(8,6))
df['Age'][df['Caesarian'] == 1].hist(color="red", edgecolor="black", bins=20, alpha=0.5, label="Caesarian = 1")
df['Age'][df['Caesarian'] == 0].hist(color="blue", edgecolor="black", bins=20, alpha=0.5, label="Caesarian = 0")
plt.legend()
plt.show()

sns.countplot(x='Heart Problem', hue='Caesarian', data=df)
plt.show()  # shows that given a heart problem your chances for caesarian section increases dramatically.

sns.countplot(x='Delivery number', hue='Caesarian', data=df)
plt.show()  # as more births you go through, the more likely it is for the next birth to be caesarian section.

sns.boxplot(x='Caesarian', y='Blood of Pressure', data=df)
plt.show()

sns.distplot(df['Blood of Pressure'], kde=False)
plt.show()

# Trying Models #
# train test split
X = df.drop('Caesarian', axis=1)
y = df['Caesarian']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=990)

# Logistic Regression #
from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()

lg_model.fit(X_train, y_train)

lg_pred = lg_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, lg_pred))
print('\n')
print(classification_report(y_test, lg_pred))

# Random Forrest #
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
              'max_features': [3,4,5],
              'max_depth': [3,5,8]}
rf_model = RandomForestClassifier()
gs = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, verbose=2)

gs.fit(X_train, y_train)
gs.best_params_

rf_pred = gs.predict(X_test)
print(confusion_matrix(y_test, rf_pred))
print('\n')
print(classification_report(y_test, rf_pred))

# XGBoost #
from xgboost import XGBClassifier
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

xgb_model = XGBClassifier()
param_grid = {'learning_rate': [0.1],
              'n_estimators': [200, 500, 1000, 2000],
              'max_depth': [3, 5, 8],
              'colsample_bytree': [0.6, 0.8, 1],
              'subsample': [0.7, 0.9, 1],
              'random_state': [209],
              'n_jobs': [2],
              'early_stopping_rounds': [10]}
gs = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, verbose=2)

gs.fit(X_train, y_train)
gs.best_params_

xgb_pred = gs.predict(X_test)
print(confusion_matrix(y_test, xgb_pred))
print('\n')
print(classification_report(y_test, xgb_pred))