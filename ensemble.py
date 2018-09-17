# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:39:17 2018

@author: xintong.yan
"""

'''
The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.
Two families of ensemble methods are usually distinguished:
* In averaging methods, the driving principle is to build several estimators independently and then to average their predictions. 
  On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.
  Examples: Bagging methods, Forests of randomized trees

* By contrast, in boosting methods, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.
  Examples: AdaBoost, Gradient Tree Boosting, …
'''


###############################   Bagging meta-estimator   ##########################################
from sklearn.datasets import make_blobs
from sklearn.ensemble import BaggingClassifier
import matplotlib as plt

X, y = make_blobs(n_samples=100, n_features=3, centers=10, cluster_std=1)



plt.pyplot.scatter(X[:,0],X[:,1], c=y)
plt.pyplot.show()

clf_bagging = BaggingClassifier(n_estimators=100,random_state=1,warm_start=True)

# fitting bagging model
clf_bagging.fit(X,y)

# predict bagging model
prediction_list = clf_bagging.predict(X)




############################# random forest ################################

'''
In extremely randomized trees (see ExtraTreesClassifier and ExtraTreesRegressor classes), randomness goes one step further in the way splits are computed.
As in random forests, a random subset of candidate features is used, but instead of looking for the most discriminative thresholds, 
thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule. 
This usually allows to reduce the variance of the model a bit more, at the expense of a slightly greater increase in bias:
'''

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib as plt

X, y = make_blobs(n_samples=10000, n_features=10, centers=10,random_state=0)
# X = data, y=label

plt.pyplot.scatter(X[:, 0], X[:, 1], c=y)
plt.pyplot.show()

# 模型1
clf_rf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
# 通过cross validation进行结果的输出
scores = cross_val_score(clf, X, y)
scores.mean()                             
# 直接输出结果
clf_rf.fit(X,y)
clf_rf.predict(X)


# 模型2
clf_rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
scores.mean()                             

# 模型3
clf_rf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
scores.mean() > 0.999



############################# Ada boosting #####################################

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_blobs
import matplotlib as plt


X, y = make_blobs(n_samples=10000, n_features=10, centers=10,random_state=0)
# X = data, y=label

plt.pyplot.scatter(X[:,0],X[:,1], c=y)

clf_adaboost = AdaBoostClassifier(n_estimators=100)

# 通过cross validation进行结果的输出
scores = cross_val_score(clf_adaboost, X, y)
scores.mean()                             

# 正常拟合
clf_adaboost.fit(X,y)
clf_adaboost.predict(X)




############################# gradient boosting #####################################
# classification

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs
import matplotlib as plt

X, y = make_blobs(n_samples=10000, n_features=10, centers=10,random_state=0)

clf_GBM = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)

clf_GBM.fit(X, y)
clf_GBM.predict(X)  























