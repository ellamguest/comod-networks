#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:15:05 2017

@author: emg
"""

import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from blockmodel_functions import *
import seaborn as sns

sub, date = 'td', '2017-10-27'

# import data
edgelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/edgelist.csv'.format(sub,date))
nodelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/nodelist.csv'.format(sub,date))
removed_subs = ['r/The_Donald','r/0','r/Not Found', 'r/changemyview']

B = bipartite_graph(edgelist, nodelist, removed_subs)

mods = moderator_network(B, edgelist)

MH, Mparititons, MBM = (build_blockmodel(mods))
mod_ei = ei_index(MH, 'type')

subs = subreddit_network(B, edgelist, removed_subs)
SH, Sparititons, SBM = (build_blockmodel(subs))

print('Output for the {} network on {}'.format(sub, date))
print ('Mod type EI index = {}'.format(np.round(mod_ei, 2)))
print('# moderator paritions - {}'.format(len(Mparititons)))
print('# subreddit paritions - {}'.format(len(Sparititons)))
print()


H, partitions, BM = build_blockmodel(subs)
draw_blockmodel(H, BM)


nm = blockmodel_df(subs)
sns.heatmap(nm)


    
 
    
        
