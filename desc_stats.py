#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:17:45 2017

@author: emg
"""

import pandas as pd

def mod_counts(edgelist, nodelist):
    removed_subs = ['r/0','r/Not Found']
    mods = nodelist[nodelist['type']==1]
    
    current = edgelist[~edgelist['sub'].isin(removed_subs)]
    active_nodes = mods[mods['name'].isin(current['name'].unique())]
    
    mod_type_counts = mods.mod_type.value_counts().sort_index()
    mod_type_counts_active = active_nodes.mod_type.value_counts().sort_index()
    
    df = pd.DataFrame({'all_mods':mod_type_counts, 'active_mods':mod_type_counts_active})
    
    df['diff'] = df['all_mods'] - df['active_mods']
    df['active_%'] = df['active_mods']/df['active_mods'].sum()
    df['all_%'] = df['all_mods']/df['all_mods'].sum()
    df['diff_%'] = df['diff']/df['diff'].sum()
    
    df['Mod Type'] = ['former non-top', 'former top', \
                      'current non-top', 'current top']
    
    return df

def inactives(edgelist):
    inactives = {'r/0': edgelist[edgelist['sub']=='r/0'].shape[0],
             'r/Not_Found': edgelist[edgelist['sub']=='r/Not Found'].shape[0]}
    
    return inactives

def mod_plots(mod_count_df):   
    mod_count_df[['all_mods', 'active_mods', 'Mod Type']].plot('Mod Type', kind='bar',
      title = '{} Mod Type Counts'.format(sub))
    mod_count_df[['all_%', 'active_%', 'Mod Type']].plot('Mod Type', kind='bar',
      title = '{} Mod Type Proportions'.format(sub))