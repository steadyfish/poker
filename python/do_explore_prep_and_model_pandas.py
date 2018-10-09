# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 18:44:49 2018

@author: Dhrumin
"""
from os import path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import pipeline as ppl
from sklearn import preprocessing as ppr

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
d_in = pd.read_csv(cdir + "\\poker\\data\\poker-hand-training-true.data", 
                   names = col_names, dtype = col_types)
# d_in = pd.read_feather(cdir +  "\\data\\poker-hand-training-true.data")
d_in.columns
d_in.shape
d_in.boxplot() # for continuous vars
d_in.describe() # for continuos vars
d_in.groupby(['s1']).count() # for categorical vars
d_in.groupby(['c1']).count() # for categorical vars
d_in.groupby(['hand']).count() # for categorical vars

# need to ensure categories are ordered
# visualize/summarise X's for each type of y(hand)
def explore_hand(d_in, hand):
    print(d_in[d_in['hand'] == hand].head())
    print(d_in[d_in['hand'] == hand].tail())

explore_hand(d_in, '0') # nothing
explore_hand(d_in, '1') # one pair
explore_hand(d_in, '2') # two pairs
explore_hand(d_in, '3') # three of a kind
explore_hand(d_in, '4') # straight
explore_hand(d_in, '5') # flush
explore_hand(d_in, '6') # full house (one pair + three of a kind)
explore_hand(d_in, '7') # four of a kind
explore_hand(d_in, '8') # straight flush
explore_hand(d_in, '9') # royal flush

# create features "pandas" way
#d_in['suit_match'] = (d_in['s1'] == d_in['s2']  == d_in['s3'] == d_in['s4'] == d_in['s5'])
d_in['suit_match'] = (d_in['s1'] == d_in['s2']) & (d_in['s1'] == d_in['s3']) & (d_in['s1'] == d_in['s4']) & (d_in['s1'] == d_in['s5'])
d_in.groupby(['suit_match']).count()
d_in.groupby(['hand', 'suit_match']).count()

# d_record = row in the dataframe
# criteria = 2, 3, 4, 5
def get_repeats(d_record, criteria):
    d_r1 = d_record.groupby(d_record).count()
    d_r2 = d_r1[d_r1 == criteria]
    return(d_r2.shape[0])

d_in['no_pairs'] = d_in[['c1', 'c2', 'c3', 'c4', 'c5']].apply(get_repeats, criteria = 2, axis = 1)
d_in['has_triplet'] = d_in[['c1', 'c2', 'c3', 'c4', 'c5']].apply(get_repeats, criteria = 3, axis = 1)
d_in['has_quartet'] = d_in[['c1', 'c2', 'c3', 'c4', 'c5']].apply(get_repeats, criteria = 4, axis = 1)
d_in.columns
d_in[['hand', 'no_pairs', 'c1']].groupby(['hand', 'no_pairs']).count()
d_in[['hand', 'has_triplet', 'c1']].groupby(['hand', 'has_triplet']).count()
d_in[['hand', 'has_quartet', 'c1']].groupby(['hand', 'has_quartet']).count()

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

d_in['has_straight'] = d_in[['c1', 'c2', 'c3', 'c4', 'c5']].apply(has_straight, axis = 1)
d_in[['hand', 'has_straight', 'c1']].groupby(['hand', 'has_straight']).count()

d_in.to_feather(cdir + "\\poker\\data\\poker-hand-training-true-precomputed.data")
d_in = pd.read_feather(cdir + "\\poker\\data\\poker-hand-training-true-precomputed.data")
d_in1 = d_in[d_in.hand.isin(['2', '3', '4', '6'])]
d_in1 = d_in

y = d_in1.hand
X = d_in1.loc[:, 'suit_match':'has_straight'] # produces a copy

estimators = [('clf', LogisticRegression())]
estimators = [('clf', RandomForestClassifier())]
pipe = ppl.Pipeline(estimators)
pipe.fit(X = X, y = y)

pipe.get_params() # model tuning parameters
pipe.named_steps['clf'].intercept_ # for LR
pipe.named_steps['clf'].coef_ # for LR
pipe.named_steps['clf'].feature_importances_ #for RF

y_pred = pipe.predict(X)
y_pred_prob = pipe.predict_proba(X)
metrics.confusion_matrix(y, y_pred)
metrics.brier_score_loss( y, y_pred_prob[:, 0])

# test
d_test = pd.read_csv(cdir + "\data\\poker-hand-testing.data", 
                     names = col_names, dtype = col_types)
d_test.columns
d_test.boxplot()

# to be able to make predictions for the test datasets, requires creating preproecssing again