import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import math
import networkx as nx
import itertools
import time
import pickle
from collections import defaultdict
from operator import itemgetter

cooffending_df = pd.read_csv('Cooffending.csv', na_values=['.'])
# print cooffending_df.loc[cooffending_df['Jeunes'] > 0].loc[cooffending_df['Adultes'] == 0]
# print cooffending_df.shape
# print cooffending_df.count('columns')
# print cooffending_df['NoUnique'].nunique()
# print cooffending_df['SeqE'].nunique()
# print cooffending_df.groupby('annee')['SeqE'].nunique()

# print cooffending_df.groupby('NCD1')['NoUnique'].nunique().sort_values(ascending=False).head(15)
# top_crime_types = [1430, 8100, 1640, 92309, 3410, 21405, 1420, 4140, 3520, 21409]
# df1 = set(cooffending_df.loc[cooffending_df['NCD1'] == "1430"]['NoUnique'])
# df2 = set(cooffending_df.loc[cooffending_df['NCD2'] == "1430"]['NoUnique'])
# df3 = set(cooffending_df.loc[cooffending_df['NCD3'] == "1430"]['NoUnique'])
# df4 = set(cooffending_df.loc[cooffending_df['NCD3'] == "1430"]['NoUnique'])
# print len(list(df1 | df2 | df3 | df4))

# the co-offending network has (weighted) adjacency matrix AAT
# A = NoUnique (rows) to SeqE (crimes)

# crimes = list(set(cooffending_df['SeqE'].unique()))
adj_list = []

# start = time.time()

# create adjacency list
for crime in crimes[:5000]:
	# group all offenders in these crimes
	d = list(cooffending_df.loc[cooffending_df['SeqE'] == crime]['NoUnique'])
	if len(d) > 1:
		# add to adjacency list
		edges = list(itertools.combinations(d, 2))
		adj_list.extend(edges)

# with open("adj.txt", "wb") as f:
# 	pickle.dump(adj_list, f)

start = time.time()

with open("adj.txt", "rb") as f:
	adj_list = pickle.load(f)

print "length of adjacency list", len(adj_list)
end = time.time()
print "time to make adj list", (end - start)

# create graph
G = nx.Graph()
G.add_edges_from(adj_list)

# get num nodes, num edges
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print "num nodes", num_nodes
print "num edges", num_edges

# solo offenders
solo_offenders = 539593 - num_nodes
print "solo offenders", solo_offenders

# number of connected components
num_cc = nx.number_connected_components(G)
print "num cc", num_cc

# nodes of largest connected components
# largest_cc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
# num_cc_nodes = largest_cc.number_of_nodes()
# print "num cc largest cc", num_cc_nodes

# save/load largest cc adj list
# print "saving cc"
# fh = open("cc.adjlist",'wb')
# nx.write_adjlist(largest_cc, fh)

start = time.time()

print "loading cc"
fh = open("cc.adjlist", 'rb')
largest_cc = nx.read_adjlist(fh)

end = time.time()
print "time to get largest cc", (end - start)

# with open("adj.txt", "wb") as f:
# 	pickle.dump(cc_adj_list, f)

# with open("adj.txt", "rb") as f:
# 	cc_adj_list = pickle.load(f)

# largest_cc = nx.Graph()
# largest_cc.add_edges_from(cc_adj_list)

def getHighestNDegree(G, n):
	return sorted(G.degree(), key=itemgetter(1), reverse=True)[:n]

def getHighestNBetweenness(G, n):
	b_centrality = nx.betweenness_centrality(G)
	tup = [(k, v) for k, v in b_centrality.iteritems()]
	return sorted(tup, key=itemgetter(1), reverse=True)[:n]

def getHighestNEigenvector(G, n):
	e_centrality = nx.eigenvector_centrality(G)
	tup = [(k, v) for k, v in e_centrality.iteritems()]
	return sorted(tup, key=itemgetter(1), reverse=True)[:n]

def plotDegreeDistribution(G):
	degrees = G.degree()
	degs = [i[1] for i in degrees]
	plt.hist(degs)
	plt.title("Degree Distribution")
	plt.xlabel("Degree")
	plt.ylabel("Number of nodes")
	plt.show()

def plotBetweennessCentrality(G):
	b_centrality = nx.betweenness_centrality(G)
	plt.hist(b_centrality.values())
	plt.title("Betweenness Centrality")
	plt.xlabel("Centrality")
	plt.ylabel("Number of nodes")
	plt.show()

def plotEigenvectorCentrality(G):
	e_centrality = nx.eigenvector_centrality(G)
	plt.hist(e_centrality.values())
	plt.title("Eigenvector Centrality")
	plt.xlabel("Centrality")
	plt.ylabel("Number of nodes")
	plt.show()

# plotDegreeDistribution(G)
# plotDegreeDistribution(largest_cc)
# print getHighestNDegree(largest_cc, 10)
# print getHighestNBetweenness(largest_cc, 10)
# print getHighestNEigenvector(largest_cc, 30)
# print nx.density(largest_cc)
# print nx.diameter(largest_cc)
# print nx.average_clustering(largest_cc)

# plotBetweennessCentrality(largest_cc)
# plotEigenvectorCentrality(largest_cc)

# plt.show()
