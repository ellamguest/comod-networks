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

def bipartite_graph(edgelist, nodelist, removed_subs):
    # create bipartitie graph
    B = nx.Graph()
    B.add_nodes_from(set(edgelist['name']), bipartite=0) 
    B.add_nodes_from(set(edgelist['sub']), bipartite=1)
    B.add_edges_from(list(zip(edgelist['name'],edgelist['sub'])))

    # add node type variable
    ## 0 sub, 1 FN, 2 FT, 3 CN, 4 CT ?
    type_dict = dict(zip(nodelist.name,nodelist.mod_type))
    nx.set_node_attributes(B, 'type', type_dict)

    # remove issue subs
    B.remove_nodes_from(removed_subs)

    return B

def subreddit_network(B, edgelist, removed_subs):
    # create unipartite subreddit graph
    subnames = [x for x in set(edgelist['sub']) if x not in removed_subs]
    subs = bipartite.weighted_projected_graph(B, (subnames))
    subs.remove_nodes_from(nx.isolates(subs))
    return subs

def moderator_network(B, edgelist):
    # create unipartite moderator graph
    mods = bipartite.weighted_projected_graph(B, edgelist['name'])
    mods.remove_nodes_from(nx.isolates(mods))
    return mods

def create_hc(G, t=1.15):
    """Creates hierarchical cluster of graph G from distance matrix"""
    # Create distance matrix
    path_length = nx.all_pairs_shortest_path_length(G)
    distances = np.zeros((len(G),len(G)))
    for u, p in path_length.items():
        for v, d in p.items():
            distances[u][v] = d            
    # Create hierarchical cluster
    Y = distance.squareform(distances)
    Z = hierarchy.complete(Y)  # Creates HC using farthest point linkage
    # This partition selection is arbitrary, for illustrative purposes
    membership = list(hierarchy.fcluster(Z, t=t))
    # Create collection of lists for blockmodel
    partition = defaultdict(list)
    for n, p in zip(list(range(len(G))),membership):
        partition[p].append(n)
    return list(partition.values())

def build_blockmodel(G):
    # Makes life easier to have consecutively labeled integer nodes
    H = nx.convert_node_labels_to_integers(G, label_attribute='label')
    
    # Create parititions with hierarchical clustering
    partitions = create_hc(H)
    
    # Build blockmodel graph
    BM = nx.blockmodel(H, partitions)
    
    BM.nodes(data=True)
    
    return H, partitions, BM


def draw_blockmodel(G, BM):
    pos = nx.spring_layout(G)
    
    fig = plt.figure(2,figsize=(8,10))
    fig.add_subplot(211)
    
    label_dict = dict([(n, G.node[n]['label']) for n in G])
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
    nx.draw(BM, posBM, node_size=node_size, width=edge_width, with_labels=True)


def blockmodel_df(G):
    H, partitions, BM = build_blockmodel(G)
    label_dict = dict([(n, H.node[n]['label']) for n in H])
    order = [label_dict[item] for sublist in partitions for item in sublist]
    nm = nx.to_pandas_dataframe(G)
    nm = nm.reindex(index = order)
    nm.columns = nm.index
    return nm


def homophilous_ties(G, attribute):
    # returns list of 1 or 0 for homophilous types between
    # modes by attribute
    homophily = {}    
    edges = G.edges()
    node_type = nx.get_node_attributes(G, attribute)
    for edge in edges:
        node1, node2 = edge
        if node_type[node1] == node_type[node2]:
            homophily[edge]=1
        else:
            homophily[edge]=0
    return homophily

def ei_index(G, attribute):
    homophily_dict = homophilous_ties(G, attribute)
    values = list(homophily_dict.values())
    EL = values.count(0)
    IL = values.count(1)   
    ei_index = (EL-IL)/(EL+IL)
    return ei_index