#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:39:50 2017

@author: emg
"""

import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

'''
DATA PROCESSING FUNCTIONS
'''
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

def print_mod_counts(sub_dict):
    df = sub_dict['mod_counts']
    print('The breakdown of mod types for {} is:'.format(sub_dict['name']))
    print()
    print(df)

def inactive_counts(edgelist):
    inactives = {'r/0': edgelist[edgelist['sub']=='r/0'].shape[0],
             'r/Not_Found': edgelist[edgelist['sub']=='r/Not Found'].shape[0]}
    
    return inactives

def network_output(edgelist, nodelist, removed_subs):
    # create bipartitie graph
    B = nx.Graph()
    B.add_nodes_from(set(edgelist['name']), bipartite=0) 
    B.add_nodes_from(set(edgelist['sub']), bipartite=1)
    B.add_edges_from(list(zip(edgelist['name'],edgelist['sub'])))

    # add node type variable
    ## 0 sub, 1 FN, 2 FT, 3 CN, 4 CT ?
    type_dict = dict(zip(nodelist.name,nodelist.mod_type))
    nx.set_node_attributes(B, type_dict, name='type')

    # remove issue subs
    B.remove_nodes_from(removed_subs)

    # create unipartite subreddit graph
    subs = bipartite.weighted_projected_graph(B,
                                              set(edgelist['sub']) - removed_subs)
    subs = subs.edge_subgraph(subs.edges())
    
    # create unipartite moderator graph
    mods = bipartite.weighted_projected_graph(B, set(edgelist['name']))
    mods = mods.edge_subgraph(mods.edges())
    
    output = {'B':B, 'mods':mods, 'subs':subs, 'type_dict':type_dict}
    
    return output

def homophily(G, attribute):
    # returns list of 1 or 0 for homophilous types between
    # modes by attribute
    homophily_dict = {}    
    edges = G.edges()
    node_type = nx.get_node_attributes(G, attribute)
    for edge in edges:
        node1, node2 = edge
        if node_type[node1] == node_type[node2]:
            homophily_dict[edge]=1
        else:
            homophily_dict[edge]=0
    
    values = list(homophily_dict.values())
    EL = values.count(0)
    IL = values.count(1)   
    ei_index = (EL-IL)/(EL+IL)
    
    return {'homophily_dict':homophily_dict,
            'ei_index':ei_index}

def blockmodel_output(G, t=1.15):
    # Makes life easier to have consecutively labeled integer nodes
    H = nx.convert_node_labels_to_integers(G, label_attribute='label')
    
    """Creates hierarchical cluster of graph G from distance matrix"""
    # Create distance matrix
    path_length = dict(nx.all_pairs_shortest_path_length(H))
    distances = np.zeros((len(H),len(H)))
    for u, p in path_length.items():
        for v, d in p.items():
            distances[u][v] = d            
    # Create hierarchical cluster
    Y = distance.squareform(distances)
    Z = hierarchy.complete(Y)  # Creates HC using farthest point linkage
    # This partition selection is arbitrary, for illustrative purposes
    membership = list(hierarchy.fcluster(Z, t=t))
    # Create collection of lists for blockmodel
    partitions = defaultdict(list)
    for n, p in zip(list(range(len(G))),membership):
        partitions[p].append(n)

    # Build blockmodel graph
    #BM = nx.blockmodel(H, partitions) # change in nx 2.0
    p_values = list(partitions.values())
    BM = nx.quotient_graph(H, p_values, relabel=True)
 
    label_dict = dict([(n, H.node[n]['label']) for n in H])
    order = [label_dict[item] for sublist in p_values for item in sublist]
    nm = nx.to_pandas_dataframe(G)
    nm = nm.reindex(index = order)
    nm.columns = nm.index

    ho = homophily(G, 'type')
    
    output = {'G':G, 'H':H, 'partitions':partitions, 'BM':BM, 'nm':nm,
              'label_dict':label_dict, 'order':order, 'distances':distances
              }
    output.update(ho)
    return output

def results_dict(sub, subname, date):
    edgelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/edgelist.csv'.format(sub,date))
    nodelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/nodelist.csv'.format(sub,date))
    removed_subs = {'r/The_Donald','r/0','r/Not Found', 'r/changemyview'}
    #removed_subs = {'test'}
    
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
    desc_table = desc_table.reindex(['# nodes','# edges', '# components',
                                     '# isolates', 'density','EI index','# BM partitions'])
    
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
        results = results_dict(sub, subname, date)
        output[sub] = results
        tables.append(results['desc_table'])

    comparison_table = pd.concat(tables, axis=1)
    comparison_table = comparison_table.reindex(['# nodes', '# edges', '# isolates', '# components', 
                    '# partitions', 'density', 'EI index'])
    
    comparison_table.columns = ['r/changemyview_mods', 'r/The_Donald_mods', 
                      'r/changemyview_subs', 'r/The_Donald_subs']
    
    output['table'] = comparison_table
    
    return output


'''
PLOTTING FUNCTIONS
'''

def mod_count_plots(sub_results):
    fig, axs = plt.subplots(1,2)
    
    df = sub_results['mod_counts']
    sub, name = sub_results['sub'], sub_results['name']
    df[['all_mods', 'active_mods', 'Mod Type']].plot('Mod Type', kind='bar',
      title = 'Mod Type Counts'.format(name), ax=axs[0])
    df[['all_%', 'active_%', 'Mod Type']].plot('Mod Type', kind='bar',
      title = 'Mod Type Proportions'.format(name), ax=axs[1])
    
    plt.tight_layout()
    
    plt.savefig('{}_mod_count_plots.png'.format(sub))
    
    plt.close()

def draw_blockmodel(H, BM, label_dict):
    pos = nx.spring_layout(H)
    
    fig = plt.figure(2,figsize=(8,10))
    fig.add_subplot(211)
    
    nx.draw(H, pos, labels=label_dict, node_size=50, with_labels=True)
    
    # Draw block model with weighted edges and nodes sized by number of internal nodes
    node_size = [BM.node[x]['nnodes']*50 for x in BM.nodes()]
    edge_width = [(2*d['weight']) for (u,v,d) in BM.edges(data=True)]
    # Set positions to mean of positions of internal nodes from original graph
    posBM = {}
    for n in BM:
        xy = np.array([pos[u] for u in BM.node[n]['graph']])
        posBM[n] = xy.mean(axis=0)
    
    fig.add_subplot(212)
    nx.draw(BM, posBM,
            node_size=node_size, width=edge_width, with_labels=True)
    
    
# attempting pretty plot
def twomode_net_plot(output, sub):
    colours = ['green', 'royalblue','midnightblue','indianred','maroon']
    cmap = LinearSegmentedColormap.from_list('Custom', colours, len(colours))
    H = output[sub]['nets']['B']    
    
    pos = nx.layout.kamada_kawai_layout(H)
    deg  = [d*20 for d in list(dict(H.degree()).values())]
    cols = list(nx.get_node_attributes(H, 'type').values())
    nx.draw(H, pos, node_size=deg, with_labels=False,
            node_color=cols, cmap=cmap, alpha=0.8)
    
    plt.savefig('twomode_net_{}.png'.format(sub))
    plt.close()
    
    
def mod_net_plot(output, sub):
    colours = ['royalblue','midnightblue','indianred','maroon']
    cmap = LinearSegmentedColormap.from_list('Custom', colours, len(colours))
    H = output[sub]['mods']['H']    
    
    pos = nx.layout.kamada_kawai_layout(H)
    deg  = [d*20 for d in list(dict(H.degree()).values())]
    cols = list(nx.get_node_attributes(H, 'type').values())
    nx.draw(H, pos, node_size=deg, with_labels=False,
            node_color=cols, cmap=cmap, alpha=0.8)
    
    plt.savefig('mod_net_{}.png'.format(sub))
    plt.close()



def sub_net_plot(output, sub):
    H = output[sub]['subs']['H']    
    
    pos = nx.layout.kamada_kawai_layout(H)
    deg  = [d*2 for d in list(dict(H.degree()).values())]
    nx.draw(H, pos, node_size=deg, with_labels=False,
            node_color='green', alpha=0.8)
    
    plt.savefig('sub_net_{}.png'.format(sub))
    plt.close()

#### TIMELINE PLOTS
### FUCNTIONS TO CONVERT MOD INSTANCES DF TO MOD PRESENCE TIMELINE
def prep_df(df):
    '''subset df into required columns and types
    to construct timeline df'''
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['pubdate'] = pd.to_datetime(df['pubdate']).dt.normalize()
    df.sort_values('pubdate', inplace=True)
    df['perm_level'] = df['permissions'].map({'+all':2}).fillna(1)
    last = df['pubdate'].max()
    n = {1:3,2:4, 0:0} 
    current = list(df[df['pubdate']==last]['name'])
    df.reset_index(inplace=True)
    c = df[df['name'].isin(current)]['perm_level'].map(n)      
    df.perm_level.update(c)     
    df.sort_values(['date','pubdate'], inplace=True)
    df.drop_duplicates(['name','date'], keep='last', inplace=True)
    df.set_index('name', inplace=True, drop=False)
    df = df[['name','date','pubdate','perm_level']]
    return df

def date_presence_dict(dates, start, end, perm_level): 
    '''check mod presence on date'''
    d = {}
    for date in dates:
        if date >= start and date <= end:
            d[date] = perm_level
    return d

def timeline_df(df):
    '''convert moderator instance date to timeline df'''
    df = prep_df(df)
    timeline = pd.DataFrame(index = pd.date_range(start = df['date'].min(),
                                                  end = df['pubdate'].max(),
                                                  freq='D'))
    for name in set(df['name']):
        if list(df['name']).count(name) == 1:
            subset = df.loc[name]
            dates = pd.date_range(start = subset['date'],
                                  end = subset['pubdate'],
                                  freq='D')
            start, end, perm_level = subset['date'], subset['pubdate'], subset['perm_level']
            d = date_presence_dict(dates, start, end, perm_level)
            timeline[name] = pd.Series(d)

        elif list(df['name']).count(name) > 1:
            combined = {}
            subset = df.loc[name]
            dates = pd.date_range(start = subset['date'].min(),
                                  end = subset['pubdate'].max(),
                                   freq='D')
            for row in subset.itertuples():
                start, end, perm_level = row[2], row[3], row[4]
                d = date_presence_dict(dates, start, end, perm_level)
                combined.update(d)
            timeline[name] = pd.Series(combined)
    timeline.fillna(0, inplace=True)
    timeline = timeline[list(df.sort_values(['date','pubdate'])['name'].drop_duplicates())]
    return timeline

####### PLOTTING FUNCTIONS
def set_cmap():
    colours = ['white','royalblue','midnightblue','indianred','maroon']
    cmap = LinearSegmentedColormap.from_list('Custom', colours, len(colours))
    return cmap
    
def td_timeline():
    td_df = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/tidy-data/td-mod-hist.csv', index_col=0)
    td_timeline = timeline_df(td_df)
    days = list(td_timeline.index)
    td_timeline.index = td_timeline.index.strftime('%Y-%m')
    
    fig = plt.figure(figsize=(15,9.27))
    ax = sns.heatmap(td_timeline, cmap=set_cmap())
   
    plt.tick_params(axis='x',which='both', labelsize=6)
    
    #plt.title('r/The_Donald Moderator Presence Timeline')
    plt.xlabel('r/The_Donald Moderators', labelpad=20)
    plt.ylabel('Moderator Presence by Date', labelpad=10)
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([1.2, 2.0, 2.8, 3.6])
    colorbar.set_ticklabels(['Former non-top',
                             'Former top',
                             'Current non-top',
                             'Current top'])
    
    plt.axhline(y=days.index(datetime(2016,11,8,0,0,0)), ls = 'dashed', color='black', label='Election')
    plt.axhline(y=days.index(datetime(2016,11,24,0,0,0)), ls = 'dotted', color='green', label='Spezgiving')
    plt.axhline(y=days.index(datetime(2017,1,21,0,0,0)), ls = 'dotted', color='black', label='Inauguration')
    plt.axhline(y=days.index(datetime(2017,5,2,0,0,0)), ls = 'dashed', color='green', label='Demodding')
    
    plt.legend(loc=9)
    
    plt.tight_layout()
    
    plt.savefig('td_mod_timeline.png', dpi=fig.dpi)
    
    plt.close()

def cmv_timeline():
    cmv_df = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/mod-list-data/cmv/history.csv', index_col=0)
    cmv_timeline = timeline_df(cmv_df)
    cmv_timeline.index = cmv_timeline.index.strftime('%Y-%m')
    
    fig = plt.figure(figsize=(8.5, 12.135))
    ax = sns.heatmap(cmv_timeline, cmap=set_cmap())
    
    #plt.title('CMV Moderator Presence Timeline', y=1.03, x=0.4, fontweight='bold')
    plt.xlabel('r/ChangeMyView Moderators',  labelpad=20)
    plt.ylabel('Moderator Presence by Date',  labelpad=10)
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([1.2, 2.0, 2.8, 3.6])
    colorbar.set_ticklabels(['Former non-top',
                             'Former top',
                             'Current non-top',
                             'Current top'])
    
    plt.tight_layout()
    
    plt.savefig('cmv_mod_timeline.png', dpi=fig.dpi)
    plt.close()



