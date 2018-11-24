# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 06:18:37 2018

@author: Dhrumin
"""

# feature engineering and model pipeline building
import os 
import numpy as np
import pandas as pd

# train test split
from sklearn.model_selection import train_test_split

# preprocessing
from sklearn import preprocessing as ppr
from sklearn import pipeline as ppl
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

# features can't be parallelly processed by different transformer in a single DataFrameMapper pipeline object
# pipeline1 also passes through 'y' (comes back as the 3rd column)
engineered_feature_pipeline1 = skp.DataFrameMapper([
        (['s1', 's2', 's3', 's4', 's5'], uf.Comparator(criteria = 5), {'alias': 'suit_match'}),
        (['c1', 'c2', 'c3', 'c4', 'c5'], uf.Comparator(criteria = 2), {'alias': 'no_pairs'})
        ], 
    input_df=True, df_out=True, default = None)

#temp = d_in[d_in['hand']=='5']
#engineered_feature_pipeline1.fit_transform(temp).head()

engineered_feature_pipeline2 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], uf.Comparator(criteria = 3), {'alias': 'has_triplet'})], 
    input_df=True, df_out=True, default = False)

engineered_feature_pipeline3 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], uf.Comparator(criteria = 4), {'alias': 'has_quartet'})], 
    input_df=True, df_out=True, default = False)

engineered_feature_pipeline4 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], uf.Straight(), {'alias': 'has_straight'})],
    input_df = True, df_out = True, default = False)

# here we lose feature names
features_pipeline = ppl.make_union(engineered_feature_pipeline1, 
                      engineered_feature_pipeline2, 
                      engineered_feature_pipeline3,
                      engineered_feature_pipeline4)

temp = d_in[d_in['hand']=='8']
#features_pipeline.fit_transform(temp).head()
a = features_pipeline.fit_transform(temp)
a[0:10, ]


# modelling complete pipeline
pipe = ppl.Pipeline([('prep', features_pipeline),
                     ('encoding', ppr.OneHotEncoder()),
                     ('clf', LogisticRegression(multi_class = 'multinomial', penalty = 'l2',
                                         random_state = 9546, solver = "lbfgs"))
#                     ('clf', RandomForestClassifier())
#                     ('sgd', SGDClassifier())
                     ])

y = d_in.hand
X = d_in.loc[:, 's1':'c5'] # produces a copy

# split - results in < 5 observations for a the smallest class (need for sampling)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, stratify = y, random_state = 9565)

# training
params_lr = dict(clf__C = [0.5, 1, 2])
estimators_rf = [('clf', RandomForestClassifier())]
log_loss_w_lbl = metrics.make_scorer(metrics.log_loss, labels = np.unique(y_train),
                                     greater_is_better = False, needs_proba = True)
cv_lr = GridSearchCV(pipe, param_grid = params_lr, scoring = log_loss_w_lbl, cv = 5) 
cv_lr.fit(X = X_train, y = y_train)

y_tr_pred = cv_lr.predict(X_train)
y_tr_pred_proba = cv_lr.predict_proba(X_train)
metrics.confusion_matrix(y_train, y_tr_pred)
metrics.brier_score_loss(y_train, y_tr_pred_proba[:, 0])

# train + test split (stratified).. done
# OneHotEncoding for categorical variables.. done
# parameter tuning for model using grid search and cross-validation.. done
# evaluation for multi-class model.. remaining
