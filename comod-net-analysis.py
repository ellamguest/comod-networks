#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:07:57 2017

@author: emg
"""

import pandas as pd
import networkx as nx
import numpy as np
from functions import *
import seaborn as sns
from networkx.algorithms import approximation
from networkx.algorithms import bipartite

date = '2017-10-27'
output = output_dict(date)


G = output['cmv']['mods']['G']

centrality = pd.DataFrame({'degrees' : nx.degree_centrality(G),
 'betweenness' : nx.betweenness_centrality(G),
 'eigenvector' : nx.eigenvector_centrality(G), 
 'closeness' : nx.closeness_centrality(G)}, dtype='category')

assert (centrality.shape == (len(G.nodes()), 4)), 'centrality table of wrong size'

centrality



import matplotlib.pyplot as plt
import pandas as pd

import prince


df = pd.read_csv('iris.csv')

pca = prince.PCA(df, n_components=4)

fig1, ax1 = pca.plot_cumulative_inertia()
fig2, ax2 = pca.plot_rows(color_by='class', ellipse_fill=True) # not working

plt.show()

'''
ATTEMPTING OWN CORRESPONDENCE ANALYSIS
from analysing social networks
the way correspondence analysis is computed is based on a singular value decomposition (SVD)
 of a normalised version of the data matrix, where the data matrix in normalised by divideing 
 each value by the square root of the product of the corresponding row and column sums
 
1) normalise data matrix
- divide each value by the square root of the product of the corresponding row and column sums

2) singular value decomposition (SVD) of the normalised data matrix
'''

from sklearn import datasets
iris = datasets.load_iris()
data = iris.data

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

X = centrality.as_matrix()

#'Data after sample-wise L2 normalizing',

N = Normalizer().fit_transform(X)

svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
svd.fit(N)  

print(svd.explained_variance_ratio_)  

print(svd.explained_variance_ratio_.sum())  
print(svd.singular_values_) 


c = nx.core_number(G)
nx.draw(G, nodelist=c.keys(), node_size=[v * 100 for v in c.values()])