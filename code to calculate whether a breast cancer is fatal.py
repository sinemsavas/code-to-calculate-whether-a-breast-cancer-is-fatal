# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:06:21 2021

@author: Sinem
"""
from sklearn.datasets import load_breast_cancer
import pandas as pd
X , y = load_breast_cancer(return_X_y=True)
df = pd.DataFrame(X, columns=load_breast_cancer().feature_names)
#looking dataset
df.head()
#statistical description
df.describe()
#We check if the data I have is empty or full.
df.isna().sum()
#mean area
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 8))
sns.distplot(df["mean area"])
#outlier detection
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df))
print(z)
outliers = list(set(np.where(z > 3)[0]))
#dropping outlier data
len(outliers)
new_df = df.drop(outliers, axis = 0).reset_index(drop = False)
display(new_df)
y_new = y[list(new_df["index"])]
#Scaling
X_new = new_df.drop('index', axis = 1)
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(X_new)
print(X_scaled)
#logistic regression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
#scaling and outlier removed
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y_new, test_size=0.3)
model = LogisticRegression()
#cross validation
cv = cross_validate(model, X_train, y_train, cv = 3, n_jobs=-1, return_estimator=True)
print(cv["test_score"])
#accuracy
print("Mean training accuracy: {}".format(np.mean(cv['test_score'])))
print("Test accuracy: {}".format(cv["estimator"][0].score(X_test, y_test)))
cv['test_score'].mean()
best_estimator_index = np.argmax(cv["test_score"])
best_estimator_model = cv["estimator"][best_estimator_index]
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
pred = best_estimator_model.predict(X_test)
cm = confusion_matrix(y_test, pred)
#visualization
plt.figure(figsize=(5, 5))
ax = sns.heatmap(cm, square=True, annot=True, cbar=False)
sns.set(font_scale=3.4)
ax.xaxis.set_ticklabels(["False","True"], fontsize = 12)
ax.yaxis.set_ticklabels(["False","True"], fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels', fontsize = 12)
ax.set_ylabel('True Labels', fontsize = 12)
plt.show()
#custom prediction 
target_names = load_breast_cancer().target_names
custom_prediction_index = int(input("Tahmin etmek istediğiniz değerin indexini girin: "))

custom_prediction = best_estimator_model.predict([X_test[custom_prediction_index]])
real_idx = y_test[custom_prediction_index]
print()
print("Estimated value:", target_names[custom_prediction][0])
print("Real value:", target_names[real_idx])


