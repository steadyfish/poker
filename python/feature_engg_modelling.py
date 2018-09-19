# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 06:18:37 2018

@author: Dhrumin
"""

# feature engineering and model pipeline building
from os import path
import pandas as pd
import numpy as np
from sklearn import pipeline as ppl
from sklearn import preprocessing as ppr
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

cdir = path.abspath("..")
col_names = ["s1", "c1", "s2", "c2", "s3", "c3", "s4", "c4", "s5", "c5", "hand"]
cat = 'category'
num = 'int32'
col_types = {"s1": cat, "c1": num, 
             "s2": cat, "c2": num, 
             "s3": cat, "c3": num, 
             "s4": cat, "c4": num, 
             "s5": cat, "c5": num, 
             "hand": cat}
d_in = pd.read_csv(cdir + "\\data\\poker-hand-training-true.data", 
                   names = col_names, dtype = col_types)
# precomputed features
d_in_pre = pd.read_feather(cdir + "\\data\\poker-hand-training-true-precomputed.data")

d_in1 = d_in_pre[d_in_pre.hand.isin(['2', '3', '4', '6'])]
d_in1 = d_in_pre

y = d_in1.hand
X = d_in1.loc[:, 'suit_match':'has_straight'] # produces a copy

estimators = [('clf', LogisticRegression())]
pipe = ppl.Pipeline(estimators)
#estimators = [('dummify', ppr.OneHotEncoder()), ('clf', SVC())]
#pipe = ppl.Pipeline(estimators)
# same as make_pipeline(ppr.OneHotEncoder(), SVC())
pipe.fit(X = X, y = y)

pipe.get_params() # model tuning parameters
pipe.named_steps['clf'].intercept_
pipe.named_steps['clf'].coef_

y_pred = pipe.predict(X)
y_pred_prob = pipe.predict_proba(X)
metrics.confusion_matrix(y, y_pred)
metrics.brier_score_loss( y, y_pred_prob[:, 1])

# temp code only
y_pred1 = pd.Series(y_pred)
pd.DataFrame([y, y_pred])
yy = pd.concat([y, y_pred1], axis = 1)#.reset_index()
yy.columns = ['hand', 'pred_hand']
yy['temp'] = 1
#pd.pivot_table(yy, index = ['hand', 'pred_hand'])
yy.groupby(['hand', 'pred_hand']).count()

## temporary
a = np.arange(0, 100, 2)
b = np.arange(100, 0, -2)
err = 2*np.random.rand(50)
c = np.exp(-(.02*a + .03*b + err))
d = 1/(1 + c)
yy = np.where(d > d.mean(), 1, 0)
xx = pd.DataFrame({'a': a, 'b':b})
# same as xx = pd.concat([a, b])

m = LogisticRegression()
m.fit(X = xx, y = yy)
m.coef_
m.intercept_
yy_pred = m.predict(xx)
yy_pred_prob = m.predict_proba(xx)
metrics.confusion_matrix(yy, yy_pred)
metrics.brier_score_loss(yy, yy_pred_prob[:, 1])

#yy_out = pd.DataFrame({'act': yy, 'pred': m.predict(xx), 'dummy': np.arange(50)})
#yy_out.groupby(['act', 'pred']).count()


