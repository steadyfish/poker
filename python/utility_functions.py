# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 07:15:49 2018

@author: Dhrumin
"""

# need to ensure categories are ordered
# visualize/summarise X's for each type of y(hand)
def explore_hand(d_in, hand):
    print(d_in[d_in['hand'] == hand].head())
    print(d_in[d_in['hand'] == hand].tail())
    

# d_record = row in the dataframe
# criteria = 2, 3, 4, 5
def get_repeats(d_record, criteria):
    d_r1 = d_record.groupby(d_record).count()
    d_r2 = d_r1[d_r1 == criteria]
    return(d_r2.shape[0])


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