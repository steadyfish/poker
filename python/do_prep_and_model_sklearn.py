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
import sklearn_pandas as skp

cdir = path.abspath(".")
col_names = ["s1", "c1", "s2", "c2", "s3", "c3", "s4", "c4", "s5", "c5", "hand"]
cat = 'category'
num = 'int32'
col_types = {"s1": cat, "c1": num, 
             "s2": cat, "c2": num, 
             "s3": cat, "c3": num, 
             "s4": cat, "c4": num, 
             "s5": cat, "c5": num, 
             "hand": cat}
d_in = pd.read_csv(cdir + "\\poker\\data\\poker-hand-training-true.data", 
                   names = col_names, dtype = col_types)
#d_in = pd.read_feather(cdir +  "\\data\\poker-hand-training-true.data")
# create features "sklearn" way

df_prep = skp.DataFrameMapper([(['s1', 's2', 's3', 's4', 's5'], my_trans_func(), {'alias': 'suit_match'}),
                               ()], 
                              input_df = True, df_out = True, default = None)
t = df_prep.fit_transform(d_in.copy())
#d_in['suit_match'] = (d_in['s1'] == d_in['s2']  == d_in['s3'] == d_in['s4'] == d_in['s5'])
d_in['suit_match'] = (d_in['s1'] == d_in['s2']) & (d_in['s1'] == d_in['s3']) & (d_in['s1'] == d_in['s4']) & (d_in['s1'] == d_in['s5'])
d_in.groupby(['suit_match']).count()
d_in.groupby(['hand', 'suit_match']).count()

d_in['suit_match1'] = d_in[['s1', 's2', 's3', 's4', 's5']].apply(get_repeats, criteria = 5, axis = 1)
d_in[['hand','suit_match','suit_match1']].groupby(['suit_match', 'suit_match1']).count()

# d_record = row in the dataframe
# criteria = 2, 3, 4, 5
def get_repeats(d_record, criteria):
    d_r1 = d_record.groupby(d_record).count()
    d_r2 = d_r1[d_r1 == criteria]
    return(d_r2.shape[0])

def get_repeats_outer(d_in, criteria):
    a = d_in.apply(get_repeats, criteria = criteria, axis = 1)
    return(a)

# example code - https://www.kaggle.com/gautham11/building-predictive-models-with-sklearn-pipelines
#from sklearn.preprocessing import FunctionTransformer

def Comparator(criteria=1):
    return ppr.FunctionTransformer(
        get_repeats_outer,
        validate=False,
        kw_args={'criteria': criteria}
    )

#d_in['has_straight'] = d_in[['c1', 'c2', 'c3', 'c4', 'c5']].apply(has_straight, axis = 1)
def has_straight(d_record):
    d_record1 = d_record.copy()
    d_record1[d_record1 == 1] = 14
    d_r1 = d_record.sort_values().diff().dropna()
    r1_all = np.all(d_r1 == 1)
    d_r2 = d_record1.sort_values().diff().dropna()
    r2_all = np.all(d_r2 == 1)
    if (r1_all) | (r2_all):
        return(1)
    else:
        return(0)

def has_straight_outer(d_in):
    a = d_in.apply(has_straight, axis = 1)
    return(a)

def Straight():
    return ppr.FunctionTransformer(
            has_straight_outer,
            validate=False)

# features can't be parallelly processed by different transformer in a single DataFrameMapper pipeline object
# pipeline4 also passes through 'y' (comes back as the 3rd column)
engineered_feature_pipeline1 = skp.DataFrameMapper([
        (['s1', 's2', 's3', 's4', 's5'], Comparator(criteria = 5), {'alias': 'suit_match'}),
        (['c1', 'c2', 'c3', 'c4', 'c5'], Comparator(criteria = 2), {'alias': 'no_pairs'})
        ], 
    input_df=True, df_out=True, default = None)

#temp = d_in[d_in['hand']=='5']
#engineered_feature_pipeline1.fit_transform(temp).head()

engineered_feature_pipeline2 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], Comparator(criteria = 3), {'alias': 'has_triplet'})], 
    input_df=True, df_out=True, default = False)

engineered_feature_pipeline3 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], Comparator(criteria = 4), {'alias': 'has_quartet'})], 
    input_df=True, df_out=True, default = False)

engineered_feature_pipeline4 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], Straight(), {'alias': 'has_straight'})],
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


# modellingcomplete pipeline
pipe = ppl.Pipeline([('prep', features_pipeline),
                     ('clf', RandomForestClassifier)])

# training
y = d_in.hand
X = d_in.loc[:, 's1':'c5'] # produces a copy

pipe.fit(X = X, y = y) # somehow preprocessing pipeline gets rid of y
y_pred = pipe.predict(X)
y_pred_proba = pipe.predict_proba(X)
metrics.confusion_metrics(y, y_pred)
metrics.brier_score_loss(y, y_pred_proba[:, 0])


