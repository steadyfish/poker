# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 07:26:14 2018

@author: Dhrumin
"""

# basic
import os 
import numpy as np
import pandas as pd

# train test split
from sklearn.model_selection import train_test_split

# preprocessing
from sklearn import preprocessing as ppr
from sklearn import pipeline as ppl
from imblearn import pipeline as imbppl
from imblearn import over_sampling as over_sampling

import sklearn_pandas as skp

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# evaluation
from sklearn import metrics

# other utility functions (assumes current working directory is 'project/python')
import utility_functions as uf

cdir = os.path.abspath("")
col_names = ["s1", "c1", "s2", "c2", "s3", "c3", "s4", "c4", "s5", "c5", "hand"]
cat = 'category'
num = 'int32'
col_types = {"s1": cat, "c1": num, 
             "s2": cat, "c2": num, 
             "s3": cat, "c3": num, 
             "s4": cat, "c4": num, 
             "s5": cat, "c5": num, 
             "hand": cat}
d_in = pd.read_csv("..\\data\\poker-hand-training-true.data", 
                   names = col_names, dtype = col_types)

# pipeline steps
# 1. feature engineering + scaling + encoding
# 2. sampling: try up-sampling, down-sampling, SMOTE from imbalanced-learn
# 3. modelling
# 4. interpreting: use lime to interpret

engineered_feature_pipeline1 = skp.DataFrameMapper([
        (['s1', 's2', 's3', 's4', 's5'], uf.Comparator(criteria = 5), {'alias': 'suit_match'}),
        (['c1', 'c2', 'c3', 'c4', 'c5'], [uf.Comparator(criteria = 2), ppr.LabelBinarizer()], {'alias': 'no_pairs'})
#        (['c1', 'c2', 'c3', 'c4', 'c5'], [uf.Comparator(criteria = 2), ppr.OneHotEncoder()], {'alias': 'no_pairs'})
        ], 
    input_df=True, df_out=True, default = False)

engineered_feature_pipeline2 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], uf.Comparator(criteria = 3), {'alias': 'has_triplet'})#,
#        (['s1', 's2', 's3', 's4', 's5'], None)
], 
    input_df=True, df_out=True, default = False)

engineered_feature_pipeline3 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], uf.Comparator(criteria = 4), {'alias': 'has_quartet'})#,
#        (['s1', 's2', 's3', 's4', 's5'], None)
], 
    input_df=True, df_out=True, default = False)

engineered_feature_pipeline4 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], uf.Straight(), {'alias': 'has_straight'})#,
#        (['s1', 's2', 's3', 's4', 's5'], None)
],
    input_df = True, df_out = True, default = False)

features_pipeline = ppl.make_union(engineered_feature_pipeline1, 
                      engineered_feature_pipeline2, 
                      engineered_feature_pipeline3,
                      engineered_feature_pipeline4)

sampling_pipeline = imbppl.make_pipeline(over_sampling.RandomOverSampler(random_state = 9565))

model_pipeline = imbppl.make_pipeline(LogisticRegression(multi_class = 'multinomial', penalty = 'l2',
                                         random_state = 9546, solver = "lbfgs"))

pipe = imbppl.Pipeline([('prep', features_pipeline),
                     ('sample', sampling_pipeline),
                     ('clf', model_pipeline)
                     ])

y = d_in.hand
X = d_in.loc[:, 's1':'c5'] # produces a copy

# split - results in < 5 observations for a the smallest class (need for sampling)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, stratify = y, random_state = 9565)

# training individual steps
X_tr_feat = features_pipeline.fit_transform(X_train, y_train)

X_tr_res, y_tr_res = sampling_pipeline.fit_resample(X_tr_feat, y_train)
np.unique(y_train, return_counts=1)
np.unique(y_tr_res, return_counts=1)

model_pipeline.fit(X = X_tr_res, y = y_tr_res)

# training
params_lr = dict(clf__C = [0.5, 1, 2])
#estimators_rf = [('clf', RandomForestClassifier())]
log_loss_w_lbl = metrics.make_scorer(metrics.log_loss, labels = np.unique(y_tr_res),
                                     greater_is_better = False, needs_proba = True)
cv_lr = GridSearchCV(model_pipeline, param_grid = params_lr, scoring = log_loss_w_lbl, cv = 5) 
cv_lr.fit(X = X_tr_res, y = y_tr_res)

pipe.fit(X = X_train, y = y_train)


y_tr_pred = cv_lr.predict(X_train)
y_tr_pred_proba = cv_lr.predict_proba(X_train)
metrics.confusion_matrix(y_train, y_tr_pred)
metrics.brier_score_loss(y_train, y_tr_pred_proba[:, 0])
