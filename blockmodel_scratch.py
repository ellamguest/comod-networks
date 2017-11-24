#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:15:05 2017

@author: emg
"""

import networkx as nx
import pandas as pd
import numpy as np
from blockmodel_functions import *
import seaborn as sns
from networkx.algorithms import approximation
from networkx.algorithms import bipartite
from tabulate import tabulate

sub, subname, date = 'cmv', 'r/changemyview', '2017-10-27'

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

def inactive_counts(edgelist):
    inactives = {'r/0': edgelist[edgelist['sub']=='r/0'].shape[0],
             'r/Not_Found': edgelist[edgelist['sub']=='r/Not Found'].shape[0]}
    
    return inactives

def results_dict(sub, subname, date):
    edgelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/edgelist.csv'.format(sub,date))
    nodelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/nodelist.csv'.format(sub,date))
    #removed_subs = {'r/The_Donald','r/0','r/Not Found', 'r/changemyview'}
    removed_subs = {'test'}
    
    nets = network_output(edgelist, nodelist, removed_subs)
    mods = blockmodel_output(nets['mods'])
    subs = blockmodel_output(nets['subs'])
    
    mods['name'], subs['name'] = '{}_mods'.format(subname), '{}_subs'.format(subname)
    networks = [mods, subs]
    
    d = {}
    for net in networks:
        G = net['G']
        d[net['name']] = {
                '# nodes': len(G.nodes()),
                '# edges': len(G.edges()),
                'density': nx.density(G),
                '# isolates': len(list(nx.isolates((G)))),
                '# components': len(list(nx.connected_components(G))),
                '# BM partitions': len(net['partitions']),
                'EI index': net['ei_index']
                 }
    
    desc_table = pd.DataFrame(d)
    
    mc = mod_counts(edgelist, nodelist)
    inactives = inactive_counts(edgelist)
    
    results = {'sub':sub, 'date':date,
               'name': subname,
               'nets':nets,
                'mods':mods,
                'subs':subs,
                'desc_table' : desc_table,
                'edgelist': edgelist,
                'nodelist':nodelist,
                'mod_counts': mc,
                'inactives': inactives}

    return results




def output_dict(date):
    sub_info = [['cmv', 'r/changemyview'], 
                ['td', 'r/The_Donald']]
    output = {}
    tables = []
    for info in sub_info:
        sub, subname = info[0], info[1]
        print('Getting output for', subname)
        results = results_dict(sub, subname, date)
        output[sub] = results
        tables.append(results['desc_table'])
        
        print()
    
    print('Getting comparison table')
    comparison_table = pd.concat(tables, axis=1)
    comparison_table = comparison_table.reindex(['# nodes', '# edges', '# isolates', '# components', 
                    '# partitions', 'density', 'EI index'])
    
    comparison_table.columns = ['r/changemyview_mods', 'r/The_Donald_mods', 
                      'r/changemyview_subs', 'r/The_Donald_subs']
    
    output['table'] = comparison_table
    
    return output

def mod_plots(sub_results):  
    df = sub_results['mod_counts']
    name = sub_results['name']
    df[['all_mods', 'active_mods', 'Mod Type']].plot('Mod Type', kind='bar',
      title = '{} Mod Type Counts'.format(name))
    df[['all_%', 'active_%', 'Mod Type']].plot('Mod Type', kind='bar',
      title = '{} Mod Type Proportions'.format(name))

#table = tabulate(output, headers=output.columns)
#print(table)
#    
#
#### ALGORITHMS
#
#
#approximation.clique.max_clique(mods['G'])
#
## independent sets
#approximation.independent_set.maximum_independent_set(mods['G'])
#approximation.independent_set.maximum_independent_set(subs['G'])
#
#from networkx.algorithms import assortativity
#
#assortativity.attribute_mixing_matrix(mods['G'], 'type') #??
#
#
###centrality measures table
#def centrality(net):
#    b = nx.betweenness_centrality(net)
#    e = nx.eigenvector_centrality(net)
#    c = nx.closeness_centrality(net)
#    d = nx.degree_centrality(net)
#    
#    data = {'betweenness':list(b.values()), 'eigenvector':list(e.values()), 
#     'closeness':list(c.values()), 'degree':list(d.values())}
#    
#    centrality = pd.DataFrame(data=data, index=list(b.keys()))
#    return centrality
#
#centrality_df = centrality(nets['subs'])
#centrality_df.mean(1).sort_values(ascending=False)
#
#
#
#### bridges
#for pair in list(nx.bridges(nets['mods'])):
#    print(nx.shortest_path(B,pair[0],pair[1]))
#
#for pair in list(nx.bridges(nets['subs'])):
#    print(nx.shortest_path(B,pair[0],pair[1]))
#    
#### hubs and authorities
#h,a=nx.hits(nets['B'])
#
### structural holes
#esize = nx.effective_size(nets['mods'])
#efficency = {n: v / nets['mods'].degree(n) for n, v in esize.items()}
#
#
#
####### VISUALS
##draw_blockmodel(mods['H'],mods['BM'],mods['label_dict'])
##draw_blockmodel(subs['H'],subs['BM'],subs['label_dict'])
##
##sns.heatmap(mods['nm'])
#
##### json for d3
##https://bl.ocks.org/mbostock/4062045 - node-link like
##  examples uses node attr 'group' - label partition as such?
# 
#from networkx.readwrite import json_graph
#import json
#
#old = nets['mods']
#G = nx.Graph()
#G.add_nodes_from(old.nodes())
#G.add_edges_from(old.edges())
#type_dict = dict(zip(nodelist.name,nodelist.mod_type.astype('str')))
#nx.set_node_attributes(G, type_dict, name='group')
#data1 = json_graph.node_link_data(G)
#
#with open('/Users/emg/Google Drive/PhD/data journalism/net-test/networkdata1.json',
#          'w') as outfile1:
#    outfile1.write(json.dumps(data1))
#
#
#        
