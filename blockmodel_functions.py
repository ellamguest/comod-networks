#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:55:22 2017

@author: emg
"""
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance

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


def draw_blockmodel(G, BM, label_dict):
    pos = nx.spring_layout(G)
    
    fig = plt.figure(2,figsize=(8,10))
    fig.add_subplot(211)
    
    nx.draw(G, pos, labels=label_dict, node_size=50, with_labels=True)
    
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



    
    


