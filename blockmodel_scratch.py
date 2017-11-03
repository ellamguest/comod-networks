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

sub, date = 'cmv', '2017-10-27'

# import data
edgelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/edgelist.csv'.format(sub,date))
nodelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/nodelist.csv'.format(sub,date))
removed_subs = {'r/The_Donald','r/0','r/Not Found', 'r/changemyview'}

nodelist.mod_type.value_counts().sort

nets = network_output(edgelist, nodelist, removed_subs)
mods = blockmodel_output(nets['mods'])
subs = blockmodel_output(nets['subs'])

def results_df(subname, mods, subs):
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
                'clu coef': approximation.clustering_coefficient.average_clustering(G),
                '# partitions': len(net['partitions']),
                'EI index': net['ei_index']
                 }
    
    results = pd.DataFrame(d)
    
    return results

def comparsion_table(date):
    subs = ['td', 'cmv']
    dfs = []
    for sub in subs:
        edgelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/edgelist.csv'.format(sub,date))
        nodelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/nodelist.csv'.format(sub,date))
        removed_subs = {'r/The_Donald','r/0','r/Not Found', 'r/changemyview'}
        
        nets = network_output(edgelist, nodelist, removed_subs)
        mods = blockmodel_output(nets['mods'])
        subs = blockmodel_output(nets['subs'])
        
        results = results_df(sub, mods, subs)
        
        dfs.append(results)
    
    output = pd.concat(dfs, axis=1)
    return output

output = comparsion_table(date)
output

table = tabulate(output, headers=output.columns)
print(table)
    

### ALGORITHMS


approximation.clique.max_clique(mods['G'])

# independent sets
approximation.independent_set.maximum_independent_set(mods['G'])
approximation.independent_set.maximum_independent_set(subs['G'])

from networkx.algorithms import assortativity

assortativity.attribute_mixing_matrix(mods['G'], 'type') #??


##centrality measures table
def centrality(net):
    b = nx.betweenness_centrality(net)
    e = nx.eigenvector_centrality(net)
    c = nx.closeness_centrality(net)
    d = nx.degree_centrality(net)
    
    data = {'betweenness':list(b.values()), 'eigenvector':list(e.values()), 
     'closeness':list(c.values()), 'degree':list(d.values())}
    
    centrality = pd.DataFrame(data=data, index=list(b.keys()))
    return centrality

centrality_df = centrality(nets['subs'])
centrality_df.mean(1).sort_values(ascending=False)



### bridges
for pair in list(nx.bridges(nets['mods'])):
    print(nx.shortest_path(B,pair[0],pair[1]))

for pair in list(nx.bridges(nets['subs'])):
    print(nx.shortest_path(B,pair[0],pair[1]))
    
### hubs and authorities
h,a=nx.hits(nets['B'])

## structural holes
esize = nx.effective_size(nets['mods'])
efficency = {n: v / nets['mods'].degree(n) for n, v in esize.items()}



###### VISUALS
#draw_blockmodel(mods['H'],mods['BM'],mods['label_dict'])
#draw_blockmodel(subs['H'],subs['BM'],subs['label_dict'])
#
#sns.heatmap(mods['nm'])

#### json for d3
#https://bl.ocks.org/mbostock/4062045 - node-link like
#  examples uses node attr 'group' - label partition as such?
 
from networkx.readwrite import json_graph
import json

old = nets['mods']
G = nx.Graph()
G.add_nodes_from(old.nodes())
G.add_edges_from(old.edges())
type_dict = dict(zip(nodelist.name,nodelist.mod_type.astype('str')))
nx.set_node_attributes(G, type_dict, name='group')
data1 = json_graph.node_link_data(G)

with open('/Users/emg/Google Drive/PhD/data journalism/net-test/networkdata1.json',
          'w') as outfile1:
    outfile1.write(json.dumps(data1))


        
