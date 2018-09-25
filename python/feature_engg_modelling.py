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
d_in = pd.read_feather(cdir +  "\\data\\poker-hand-training-true.data")
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

# example code - https://www.kaggle.com/gautham11/building-predictive-models-with-sklearn-pipelines
#from sklearn.preprocessing import FunctionTransformer

def Comparator(criteria=1):
#def ColumnsEqualityChecker(result_column='equality_col', inverse=False):
    def get_repeats_outer(d_in, criteria):
        
        def get_repeats(d_record, criteria):
#            print(type(d_record))
#            print(d_record.shape)
            d_r1 = d_record.groupby(d_record).count()
            d_r2 = d_r1[d_r1 == criteria]
            return(d_r2.shape[0])
        a = d_in[['s1', 's2', 's3', 's4', 's5']].apply(get_repeats, criteria = criteria, axis = 1)
        return(a)
#        return pd.DataFrame(eq.values.astype(int), columns=[result_column])
    return ppr.FunctionTransformer(
        get_repeats_outer,
        validate=False,
        kw_args={'criteria': criteria}
#        kw_args={'result_column': result_column, 'inverse': inverse}
    )

engineered_feature_pipeline1 = skp.DataFrameMapper([
        (['s1', 's2', 's3', 's4', 's5'], Comparator(criteria = 5), {'alias': 'suit_match'}),
        (['c1', 'c2', 'c3', 'c4', 'c5'], Comparator(criteria = 2), {'alias': 'no_pairs'}),
        (['c1', 'c2', 'c3', 'c4', 'c5'], Comparator(criteria = 3), {'alias': 'has_triplet'}),
        (['c1', 'c2', 'c3', 'c4', 'c5'], Comparator(criteria = 4), {'alias': 'has_quartet'})], 
    input_df=True, df_out=True, default = None)

temp = d_in[d_in['hand']=='2']
engineered_feature_pipeline1.fit_transform(temp).head()

engineered_feature_pipeline2 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], Comparator(criteria = 3), {'alias': 'has_triplet'})], 
    input_df=True, df_out=True, default = None)

engineered_feature_pipeline3 = skp.DataFrameMapper([
        (['c1', 'c2', 'c3', 'c4', 'c5'], Comparator(criteria = 4), {'alias': 'has_quartet'})], 
    input_df=True, df_out=True, default = None)

features_pipeline = make_union(engineered_feature_pipeline1, 
                      engineered_feature_pipeline2, 
                      engineered_feature_pipeline3)

temp = d_in[d_in['hand']=='2']
features_pipeline.fit_transform(temp).head()
